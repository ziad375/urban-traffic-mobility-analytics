# NYC Traffic & Mobility Analytics

A Big Data analytics project for processing and analyzing large-scale NYC taxi trip data using a distributed data processing pipeline.

The project demonstrates an end-to-end Big Data workflow including data ingestion, distributed storage, ETL processing, workflow orchestration, analytical storage, and interactive visualization.

---

## Project Overview

The goal of this project is to build a scalable data pipeline for NYC taxi trip data and generate useful mobility insights such as:

- Total number of trips
- Total revenue
- Average trip speed
- Average trip duration
- Trips by hour
- Daily trip trends
- Top pickup zones by number of trips
- Top pickup zones by revenue
- Geographic zone analysis using NYC Taxi Zone names

The project uses Docker to run the main Big Data components in an integrated environment.

---

## Architecture

```text
                 NYC Taxi Data
                       |
                       v
                 +-----------+
                 |   HDFS    |
                 | Raw Data  |
                 +-----------+
                       |
                       v
                 +-----------+
                 |   Spark   |
                 |    ETL    |
                 +-----------+
                       |
                       v
              Processed Parquet Data
                       |
             +---------+---------+
             |                   |
             v                   v
       +-----------+       +-----------+
       | ClickHouse|       |  HDFS     |
       | Analytics |       | Processed |
       +-----------+       +-----------+
             |
             v
       +-------------+
       |  Superset   |
       | Dashboard   |
       +-------------+

        Airflow orchestrates
        the ETL workflow

## Data Pipeline

### 1. Data Storage

Raw NYC taxi data is stored in HDFS:

```text
/data/nyc/raw
```

Processed data is stored separately:

```text
/data/nyc/processed
```

This separation keeps the raw data available while allowing the processed datasets to be used by downstream analytical systems.

### 2. Apache Spark ETL

Apache Spark performs the main ETL operations.

The ETL process includes:

- Reading raw Parquet data from HDFS
- Cleaning and transforming the data
- Preparing trip-level data
- Generating hourly zone-level aggregations
- Writing processed data back to HDFS

Main ETL scripts:

```text
etl_clean.py
etl_clean_staging.py
```

### 3. Apache Airflow

Airflow is used to orchestrate the data pipeline.

The main DAG is:

```text
airflow/dags/nyc_traffic_pipeline.py
```

The workflow contains the following stages:

```text
check_hdfs
     |
     v
check_spark
     |
     v
preflight_staging
     |
     v
run_etl
     |
     v
validate_staging
```

This ensures that the required infrastructure is available before running the ETL process and validates the generated output afterward.

---

## ClickHouse Analytics

ClickHouse is used as the analytical database for fast querying of the processed data.

Main analytical table:

```text
default.zone_hour
```

The dataset contains zone-level and hourly aggregated information including:

- Pickup zone
- Pickup date
- Pickup hour
- Number of trips
- Average fare
- Average distance
- Average trip duration
- Average speed
- Total revenue

---

## NYC Taxi Zone Lookup

The project uses a Taxi Zone lookup table to convert numeric pickup zone IDs into readable NYC zone names.

```text
LocationID → Zone Name
```

The lookup data is stored in:

```text
taxi_zone_lookup.csv
```

This allows the dashboard to display meaningful zone names instead of numeric IDs.

---

## Superset Dashboard

Apache Superset is used to build an interactive analytics dashboard.

### Dashboard KPIs

The dashboard provides:

- Total Trips
- Total Revenue
- Average Speed
- Average Trip Duration

### Visualizations

The dashboard includes:

- Daily Trips Trend
- Trips by Hour
- Top 10 Pickup Zones by Trips
- Top 10 Zones by Revenue

The top-zone visualizations use NYC Taxi Zone names for easier interpretation.

### Dashboard Filters

Interactive filters include:

- Time Range
- Pickup Hour
- Pickup Zone

---

## Running the Project

### Prerequisites

Make sure the following are installed:

- Docker Desktop
- Git
- Git Bash / Command Prompt / PowerShell

Recommended system resources depend on the size of the NYC taxi dataset being processed.

### Clone the Repository

```bash
git clone https://github.com/ziad375/urban-traffic-mobility-analytics.git
cd urban-traffic-mobility-analytics
```

### Start the Environment

```bash
docker compose up -d
```

Check the running containers:

```bash
docker compose ps
```

---

## Main Services

| Service | Port | Purpose |
|---|---:|---|
| HDFS NameNode | 9870 | HDFS Web UI |
| Spark Master | 8080 | Spark Cluster UI |
| Spark Worker | 8081 | Spark Worker UI |
| Airflow | 8085 | Workflow management |
| Superset | 8088 | Analytics dashboard |
| ClickHouse | 8123 | Analytical database |

---

## Airflow

Open the Airflow interface:

```text
http://localhost:8085
```

The main DAG is:

```text
nyc_traffic_pipeline
```

Trigger the DAG from the Airflow UI to execute the pipeline.

---

## Spark

Spark Master UI:

```text
http://localhost:8080
```

Spark Worker UI:

```text
http://localhost:8081
```

---

## HDFS

NameNode Web UI:

```text
http://localhost:9870
```

Main HDFS locations:

```text
/data/nyc/raw
/data/nyc/processed
/data/nyc/staging
```

---

## Superset

Open:

```text
http://localhost:8088
```

The main dashboard is:

```text
NYC Traffic Analytics
```

---

## Example Analytics

The dashboard can be used to answer questions such as:

- How many taxi trips were recorded?
- Which pickup zones have the highest trip volume?
- Which zones generate the most revenue?
- What hours have the highest traffic?
- How does trip volume change over time?
- What is the average trip speed?
- What is the average trip duration?

---

## Machine Learning & Analytics

The repository also contains scripts related to additional analytics and machine learning experiments, including:

```text
sparkml_clustering.py
train_xgboost.py
train_trip_models.py
train_trip_models_no_distance.py
```

These scripts support further analysis of NYC taxi mobility patterns and trip-related predictions.

---

## Key Features

- Distributed storage using HDFS
- Distributed processing using Apache Spark
- Automated ETL orchestration with Airflow
- Analytical querying with ClickHouse
- Interactive visualization using Apache Superset
- NYC Taxi Zone name mapping
- Dockerized Big Data environment
- Reproducible data pipeline
- Git-based version control

---

## Future Improvements

Possible future improvements include:

- Real-time streaming using Kafka
- Real-time dashboards
- More advanced machine learning models
- Geographic visualization using NYC maps
- Automated model retraining
- Cloud deployment
- Additional traffic and mobility datasets

---

## Author

**Farouk Sameh Mostafa**

**Mostafa Ahmed**

**Ziad Sherif**

**Gaber Tahoon**

---

## Repository

GitHub:

https://github.com/ziad375/urban-traffic-mobility-analytics
