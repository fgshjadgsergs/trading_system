# Торговая система «Карта ликвидности»

Реализация по чеклисту `checklist-karta-likvidnosti.md`. Binance USDT-M — первая биржа;
схема данных и интерфейс `ExchangeAdapter` мульти-биржевые с первого дня.

## Структура

```
trading_system/
  core/        — единая схема (exchange, symbol, ts_event, ts_recv; UTC ns; объёмы в монетах и USD),
                 интерфейс ExchangeAdapter (subscribe, normalize, snapshot, liq_formula),
                 parquet-хранилище (партиции exchange/symbol/date, ротация по часам),
                 конфиги yaml, фиксированные сиды, синтетические данные для тестов
  collectors/  — M1: WS depth@100ms / aggTrade / forceOrder / markPrice / kline_1m,
                 REST openInterest и ратио, реконнект, детект гэпов, батч-запись;
                 загрузка истории с data.binance.vision
  book/        — M2: реконструктор стакана (снапшот + дифы по U/u/pu), реплей
  features/    — M3: тайм/объёмные бары из тиков, дельта, CVD, ΔOI asof-join, VWAP, ATR,
                 мульти-ТФ фичи (z-объёмы 1м…1д, импульсные бары, скорость ΔOI)
  liqmap/      — M4: карта ликвидаций (liq_price, allocate, update, consume, decay)
  profile/     — M5: профиль объёма, POC/VA/HVN/LVN, свинги, стоп-кластеры
  spoof/       — M6: эвристики спуфинга по жизни L2-уровней, stability score
  signals/     — M7: S1 «магнит», S2 «свип-разворот», S3 фильтр
  backtest/    — M8: событийный бэктестер (fill-модель, задержка, комиссии, funding)
  risk/        — M9: сайзинг по воле, дневной стоп, kill switch, стейт-машина ордера
  viz/         — M10: единый стиль; seaborn — аналитика, mplfinance/plotly — свечи и overlay
  monitoring/  — M11: свежесть потоков, алерты, live-PnL против бэктеста
  calibration/ — этап 3: event studies, калибровка весов плеч, walk-forward
configs/       — yaml-конфиги
scripts/       — run_pipeline.py (сквозной прогон), загрузка Vision, отчёт качества
tests/         — pytest + hypothesis
reports/       — png-отчёты каждого модуля
```

## Запуск

```bash
pip install -e ".[dev]"
pytest                          # все тесты
python scripts/run_pipeline.py  # сквозной прогон на синтетике/записях
```

Все случайности сидированы (`configs/base.yaml: project.seed`). CI — GitHub Actions
(`.github/workflows/ci.yml`): ruff + pytest.
