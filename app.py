import streamlit as st
import pandas as pd
import xgboost as xgb
from datetime import date, time

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="NYC Taxi Trip Predictor",
    page_icon="🚕",
    layout="centered"
)

@st.cache_data
def load_zones():
    zones = pd.read_csv(
    "taxi_zone_lookup.csv",
    header=None,
    names=["LocationID", "Borough", "Zone", "service_zone"]
         )
    zones["LocationID"] = zones["LocationID"].astype(int)
    zones["Zone"] = zones["Zone"].astype(str)
    zones["Borough"] = zones["Borough"].astype(str)

    zones["display_name"] = (
        zones["Zone"] + " — " + zones["Borough"]
        + " (ID: " + zones["LocationID"].astype(str) + ")"
    )
    return zones

zones = load_zones()


# Page configuration
st.set_page_config(
    page_title="NYC Taxi Trip Predictor",
    page_icon="🚕",
    layout="centered"
)

st.title("NYC Taxi Trip Predictor")
st.write(
    "Predict taxi trip duration and estimated total amount "
    "using XGBoost models trained on NYC taxi trip data."
)

# Load trained models without scikit-learn
@st.cache_resource
def load_models():
    duration_model = xgb.Booster()
    duration_model.load_model(
        "xgboost_trip_models_no_distance/duration_min_xgboost.json"
    )

    amount_model = xgb.Booster()
    amount_model.load_model(
        "xgboost_trip_models_no_distance/total_amount_xgboost.json"
    )

    return duration_model, amount_model


try:
    duration_model, amount_model = load_models()
except Exception as e:
    st.error(f"Could not load the models: {e}")
    st.stop()

# User inputs
st.subheader("Trip Information")

col1, col2 = st.columns(2)

zone_options = zones["display_name"].tolist()
zone_ids = zones["LocationID"].tolist()

with col1:
    pickup_name = st.selectbox(
        "Pickup Zone",
        zone_options,
        index=zone_options.index(
            zones.loc[zones["LocationID"] == 161, "display_name"].iloc[0]
        ) if (zones["LocationID"] == 161).any() else 0
    )

with col2:
    dropoff_name = st.selectbox(
        "Drop-off Zone",
        zone_options,
        index=zone_options.index(
            zones.loc[zones["LocationID"] == 237, "display_name"].iloc[0]
        ) if (zones["LocationID"] == 237).any() else 0
    )

pickup_zone = int(
    zones.loc[zones["display_name"] == pickup_name, "LocationID"].iloc[0]
)

dropoff_zone = int(
    zones.loc[zones["display_name"] == dropoff_name, "LocationID"].iloc[0]
)
trip_date = st.date_input(
    "Pickup Date",
    value=date.today(),
    min_value=date(2022, 1, 1),
    max_value=date(2026, 12, 31)
)

trip_time = st.time_input(
    "Pickup Time",
    value=time(14, 0)
)

# Convert Python weekday to Spark dayofweek
# Spark: Sunday=1, Saturday=7
spark_dayofweek = ((trip_date.weekday() + 1) % 7) + 1
is_weekend = int(spark_dayofweek in [1, 7])

# Prepare model features
feature_order = [
    "pu_zone",
    "do_zone",
    "pickup_hour",
    "day_of_week",
    "month",
    "year",
    "is_weekend"
]

features = pd.DataFrame([{
    "pu_zone": int(pickup_zone),
    "do_zone": int(dropoff_zone),
    "pickup_hour": trip_time.hour,
    "day_of_week": spark_dayofweek,
    "month": trip_date.month,
    "year": trip_date.year,
    "is_weekend": is_weekend
}])[feature_order]

# Prediction
if st.button("Predict Trip", type="primary"):
    try:
        dtest = xgb.DMatrix(
            features,
            feature_names=feature_order
        )

        duration = float(duration_model.predict(dtest)[0])
        amount = float(amount_model.predict(dtest)[0])

        duration = max(0.0, duration)
        amount = max(0.0, amount)

        st.subheader("Prediction Results")

        result1, result2 = st.columns(2)

        with result1:
            st.metric(
                "Estimated Duration",
                f"{duration:.1f} minutes"
            )

        with result2:
            st.metric(
                "Estimated Total Amount",
                f"${amount:.2f}"
            )

        st.caption(
            "Predictions are estimates based on historical data, "
            "not guaranteed trip outcomes. Use zone IDs represented "
            "in the training data."
        )

        st.subheader("Input Features")
        st.dataframe(features, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")