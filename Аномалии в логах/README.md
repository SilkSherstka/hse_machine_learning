# Log Anomaly Detector

Проект для детекции аномалий в веб-трафике на датасете `web_traffic.csv`:
- baseline: `Z-score` и `IQR`
- улучшенная модель: `Isolation Forest`

Подробная архитектура и схема data flow: [`docs/architecture.md`](docs/architecture.md)

## Структура

```text
log-anomaly-detector/
├── README.md
├── docs/
│   └── architecture.md
├── requirements.txt
├── config.yaml
├── web_traffic.csv
├── baseline_traffic_anomaly.ipynb
├── models/
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── feature_engineering.py
│   └── models/
│       ├── baselines.py
│       ├── isolation_forest.py
│       ├── evaluator.py
│       ├── train.py
│       └── predict.py
```

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.models.train --data_path web_traffic.csv
python -m src.models.predict --data_path web_traffic.csv
```

После обучения артефакты сохраняются в `models/`:
- `isolation_forest.joblib`
- `score_threshold.joblib`
- `feature_columns.joblib`
