# NYC Traffic & Mobility Analytics

An end-to-end Big Data project on NYC yellow-taxi trips. Raw trip data is stored in **HDFS**, cleaned and aggregated with **Spark**, orchestrated by **Airflow**, explored in **Superset** dashboards (via **ClickHouse**), and used to train **XGBoost** models that power a **Streamlit** trip predictor.

## What you get

- **Dashboard** (Superset): total trips, total revenue, average speed and duration, daily trends, trips by hour, top pickup zones by trips and by revenue, with filters for time range, hour and zone.
- **Trip predictor** (Streamlit): estimates the duration and total fare of a trip from pickup/drop-off zone, distance, passengers, date and time.
- **Forecasting models**: XGBoost models for trips and revenue per pickup zone and hour.
- **Data-quality report**: the ETL prints how many rows break each cleaning rule before removing them.

## Architecture

```text
                  NYC taxi Parquet files (local)
                              |
                              v
                     +------------------+
                     |  HDFS /raw       |
                     +------------------+
                              |
        Airflow  ------>      v
     (orchestrates)    +------------------+
                       |  Spark ETL       |   etl/etl_clean.py
                       +------------------+
                              |
                              v
                 HDFS: trips/  and  zone_hour/
                              |
          +-------------------+-------------------+
          |                                       |
          v                                       v
   +-------------+                      +--------------------+
   | ClickHouse  |                      | XGBoost training   |  ml/
   | zone_hour   |                      +--------------------+
   +-------------+                                |
          |                                       v
          v                               models/*.json
   +-------------+                                |
   |  Superset   |                                v
   |  dashboard  |                        +---------------+
   +-------------+                        | Streamlit app |  app/app.py
                                          +---------------+
```

## Project structure

```text
.
├── docker-compose.yml        # HDFS, Spark, Airflow (+Postgres), ClickHouse, Superset
├── airflow/
│   ├── Dockerfile            # Airflow image with Java + PySpark
│   └── dags/
│       └── nyc_traffic_pipeline.py   # DAG: checks -> ETL -> validation
├── hadoop-config/            # core-site.xml, hdfs-site.xml shared by the containers
├── superset/
│   ├── Dockerfile            # Superset image with the ClickHouse driver
│   └── .env.example          # copy to .env before starting
├── etl/
│   └── etl_clean.py          # Spark ETL: raw -> clean trips + zone/hour aggregates
├── ml/
│   ├── train_trip_models.py       # trip duration + fare models (--no-distance variant)
│   └── train_zone_hour_models.py  # trips + revenue forecasting per zone/hour
├── app/
│   ├── app.py                # Streamlit trip predictor
│   └── requirements.txt
├── models/                   # trained XGBoost models + metrics.json
│   ├── trip_models/                # used by the app
│   ├── trip_models_no_distance/
│   └── zone_hour_models/
├── data/
│   └── taxi_zone_lookup.csv  # LocationID -> borough / zone name
├── docs/
│   ├── dashboard/            # exported Superset dashboard (PDF + image)
│   └── presentation/         # project slides
├── original/                 # files from before they were merged (see below)
└── unused/                   # scripts not used by the pipeline
```

## Services and ports

| Service | URL / Port | Purpose |
|---|---|---|
| HDFS NameNode | http://localhost:9870 (RPC 8020) | HDFS web UI |
| HDFS DataNode | 9864 | Block storage |
| Spark Master | http://localhost:8080 (7077) | Cluster UI / job submission |
| Spark Worker | http://localhost:8081 | Worker UI (4 GB, 2 cores) |
| Airflow | http://localhost:8085 | Workflow UI (user `admin`, password `admin`) |
| Superset | http://localhost:8088 | Dashboards (credentials from `superset/.env`) |
| ClickHouse | http://localhost:8123 (native 9000) | Analytical database |

## Getting started

**Prerequisites:** Docker Desktop and Git. The Spark worker is configured for 4 GB of memory, so give Docker at least that much plus room for the other containers.

### 1. Clone and configure

```bash
git clone https://github.com/ziad375/urban-traffic-mobility-analytics.git
cd urban-traffic-mobility-analytics
```

Create the Superset settings file (it is git-ignored) and edit the password and secret key:

```bash
cp superset/.env.example superset/.env      # Windows PowerShell: copy superset\.env.example superset\.env
```

### 2. Start the stack

```bash
docker compose up -d
docker compose ps
```

### 3. Load the raw data into HDFS

The raw Parquet files are not stored in the repository. Put the monthly NYC yellow-taxi files (`yellow_tripdata_YYYY-MM.parquet`) in `/data/nyc/raw` in HDFS, for example:

```bash
docker exec namenode hdfs dfs -mkdir -p /data/nyc/raw
docker cp <your-local-parquet-folder>/. namenode:/tmp/raw
docker exec namenode sh -c "hdfs dfs -put /tmp/raw/*.parquet /data/nyc/raw/"
```

### 4. Run the ETL

There are two ways to run `etl/etl_clean.py`. Both read `/data/nyc/raw` and write two datasets, `trips/` and `zone_hour/`.

**Through Airflow (recommended for a tracked run).** Open http://localhost:8085, enable and trigger the `nyc_traffic_pipeline` DAG. It runs these steps in order:

```text
check_hdfs -> check_spark -> preflight_staging -> run_etl -> validate_staging
```

Each run writes to its own folder, `/data/nyc/staging/runs/<run_id>/`, so earlier results are never overwritten.

**Directly with spark-submit.** With no `ETL_OUTPUT_BASE` set, the output goes to `/data/nyc/processed/`, which is where the ML scripts read from:

```bash
docker exec airflow spark-submit \
  --master spark://spark-master:7077 --deploy-mode client \
  --conf spark.driver.host=airflow --conf spark.driver.bindAddress=0.0.0.0 \
  /opt/airflow/etl_clean.py
```

Rows are removed by the ETL when they have missing zones or timestamps, a pickup year outside 2021-2026, a duration outside 1-180 minutes, a distance outside 0-100 miles, a non-positive fare, a passenger count outside 1-6, or an average speed above 80 mph.

## Dashboard (ClickHouse + Superset)

Superset reads the `zone_hour` aggregate from ClickHouse (table `default.zone_hour`). Its columns are `pu_zone`, `pickup_date`, `pickup_hour`, `trips`, `avg_fare`, `avg_distance`, `avg_duration_min`, `avg_speed_mph` and `total_revenue`.

> Loading `zone_hour` from HDFS into ClickHouse and building the Superset charts is done manually; it is not scripted in this repository.

`data/taxi_zone_lookup.csv` maps the numeric `LocationID` to a zone and borough name, so charts can show names such as "Midtown Center" instead of IDs. A snapshot of the finished dashboard is in `docs/dashboard/`.

## Machine learning

Both training scripts hold out time: they train on data before **2025-10-01** and test on data from that date onward (randomly sampled to fit in memory).

| Model | Script | Features | Test metrics |
|---|---|---|---|
| Trip duration (min) | `ml/train_trip_models.py` | zones, passengers, distance, hour, weekday, month, year, weekend | MAE 4.16, RMSE 7.17, R² 0.79 |
| Trip total amount ($) | `ml/train_trip_models.py` | same as above | MAE 3.82, RMSE 7.10, R² 0.90 |
| Duration, no distance | `ml/train_trip_models.py --no-distance` | zones, hour, weekday, month, year, weekend | MAE 6.39, RMSE 10.42, R² 0.56 |
| Total amount, no distance | `ml/train_trip_models.py --no-distance` | same as above | MAE 6.79, RMSE 11.09, R² 0.76 |
| Trips per zone-hour | `ml/train_zone_hour_models.py` | zone, calendar fields, previous hour's trips/revenue, hours since previous | MAE 5.55, RMSE 13.60, R² 0.96 |
| Revenue per zone-hour | `ml/train_zone_hour_models.py` | same as above | MAE 192.27, RMSE 620.76, R² 0.94 |

Metrics are the values stored in each `models/*/metrics.json`.

To retrain, run the script in the Spark container (the driver needs `xgboost`, `pandas` and `numpy`; install them there if they are missing) and copy the result into `models/`:

```bash
docker cp ml/train_trip_models.py spark-master:/tmp/
docker exec spark-master spark-submit --master spark://spark-master:7077 /tmp/train_trip_models.py
docker cp spark-master:/tmp/xgboost_trip_models/. models/trip_models/
```

| Script output (inside the container) | Copy to |
|---|---|
| `/tmp/xgboost_trip_models` | `models/trip_models` |
| `/tmp/xgboost_trip_models_no_distance` | `models/trip_models_no_distance` |
| `/tmp/xgboost_models` | `models/zone_hour_models` |

## Trip predictor app

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

The app uses the models in `models/trip_models/` and the zone names from `data/taxi_zone_lookup.csv`. It can be started from any folder.

## Other folders

- `original/` holds the files as they were before this reorganization: `etl_clean.py` and `etl_clean_staging.py` (merged into `etl/etl_clean.py`), `train_trip_models.py` and `train_trip_models_no_distance.py` (merged into `ml/train_trip_models.py`), and the previous `README_old.md`.
- `unused/` holds `prepare_xgboost_data.py`, which only prints the train/test split and writes nothing.

## Future improvements

- Real-time streaming with Kafka and live dashboards
- Geographic visualization using NYC maps
- Automated model retraining
- Cloud deployment
- Additional traffic and mobility datasets

## Authors

Farouk Sameh Mostafa, Mostafa Ahmed, Ziad Sherif, Gaber Tahoon

Repository: https://github.com/ziad375/urban-traffic-mobility-analytics
