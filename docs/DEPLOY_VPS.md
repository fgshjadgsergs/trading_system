# Развёртывание live-рекордера на VPS

Зачем VPS: (1) с сети записи Binance отдаёт только depth — сделки/маркпрайс/клайны/ликвидации
глушатся на стороне шлюза (проверено `ws_probe`: подписка подтверждается, данные не идут);
(2) Gate 1 требует 7 дней записи без немаркированных гэпов — домашний ПК с его сном и
перезагрузками для этого не подходит. Любой KVM-VPS за ~$5/мес в незаблокированном регионе
(Нидерланды, Германия, Сингапур, Япония…) решает обе проблемы.

Требования: Ubuntu 22.04/24.04, 1 vCPU, 1–2 ГБ RAM, диск 25 ГБ
(лейк по трём символам ≈ 0.5–1 ГБ/день в сжатом parquet).

## 1. Установка (один раз, ~5 минут)

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv git unzip
# код: либо git clone своей копии репозитория, либо закинуть zip через scp
cd ~ && unzip trading_system_src.zip -d trading_system && cd trading_system
python3.11 -m venv .venv && .venv/bin/pip install -e .
```

Проверка, что с этого сервера потоки реально идут (главный тест перед подпиской на месяц —
некоторые хостинги тоже в списках Binance; при нулях по markPrice меняйте регион/хостера):

```bash
.venv/bin/python scripts/ws_probe.py --mode url --kinds mark_price --seconds 20
.venv/bin/python scripts/ws_probe.py --seconds 15
```

В `ИТОГ:` должны быть ненулевые `markPriceUpdate`, `aggTrade`, `kline_1m` (forceOrder редкий).

## 2. Рекордер как systemd-сервис (авто-рестарт и старт после ребута)

```bash
sudo cp deploy/trading-recorder.service /etc/systemd/system/
# при иных пути установки/пользователе поправьте User= и пути внутри юнита
sudo systemctl daemon-reload
sudo systemctl enable --now trading-recorder
```

Наблюдение:

```bash
journalctl -u trading-recorder -f            # живой лог (heartbeat stream_counts раз в 60 с)
systemctl status trading-recorder            # состояние/аптайм
du -sh ~/trading_system/data/live_lake       # рост лейка
```

В `stream_counts` должны присутствовать `DepthDiff`, `Trade`, `MarkPrice`, `Kline`,
`OpenInterest`, `RatioPoint`, со временем — `Liquidation`.

## 3. Ежедневный контроль качества (Gate 1)

```bash
cd ~/trading_system
for s in ETHUSDT BTCUSDT SOLUSDT; do
  .venv/bin/python scripts/quality_report.py --lake data/live_lake --symbol "$s"
done
```

Отчёт печатает аптайм по потокам и список гэпов; немаркированные гэпы = провал Gate 1.

## 4. Калибровка Gate A (после 2–3 дней записи)

Либо прямо на VPS:

```bash
.venv/bin/python scripts/stage3_report.py --lake data/live_lake --skip-download \
    --symbol ETHUSDT --embargo-days 0.25 --test-frac 0.3
```

Либо забрать лейк к себе и запустить локально:

```bash
# на своей машине (пример для Windows PowerShell + scp из OpenSSH)
scp -r user@VPS_IP:~/trading_system/data/live_lake .\data\live_lake
python scripts\stage3_report.py --lake data\live_lake --skip-download --symbol ETHUSDT --embargo-days 0.25 --test-frac 0.3
```

Отчёт с вердиктом Gate A появится в `reports/stage3-ethusdt/README.md`.
