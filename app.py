
import streamlit as st
import pandas as pd
import xgboost as xgb
from datetime import date, time

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="NYC Mobility | Trip Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CUSTOM DESIGN ====================
st.markdown("""
<style>
    .stApp {
        background: #0b1120;
        color: #e5e7eb;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, 0.9);
    }

    .hero {
        background: linear-gradient(120deg, #172554, #111827 70%);
        border: 1px solid #293548;
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 24px;
    }

    .eyebrow {
        color: #60a5fa;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #f8fafc;
        font-size: 2.35rem;
        margin: 10px 0;
    }

    .hero p {
        color: #b8c4d6;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .section-heading {
        color: #f1f5f9;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 24px 0 14px 0;
    }

    .panel {
        background: #111a2b;
        border: 1px solid #263449;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 16px;
    }

    .result-label {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }

    .result-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 750;
        line-height: 1.3;
    }

    .result-unit {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 5px;
    }

    .route-card {
        background: #111a2b;
        border: 1px solid #263449;
        border-radius: 14px;
        padding: 18px 20px;
        margin-top: 14px;
        margin-bottom: 14px;
    }

    .route-label {
        color: #94a3b8;
        font-size: 0.78rem;
        margin-bottom: 5px;
    }

    .route-value {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 0.96rem;
    }

    div.stButton > button[kind="primary"] {
        background: #2563eb;
        border: 1px solid #3b82f6;
        color: white;
        border-radius: 10px;
        min-height: 48px;
        font-weight: 700;
        transition: 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #1d4ed8;
        border-color: #60a5fa;
    }

    div[data-testid="stMetric"] {
        background: #111a2b;
        border: 1px solid #263449;
        border-radius: 14px;
        padding: 18px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 38px;
        padding-top: 18px;
        border-top: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)


# ==================== DATA ====================
@st.cache_data
def load_zones():
    zones = pd.read_csv(
        "taxi_zone_lookup.csv",
        header=None,
        names=["LocationID", "Borough", "Zone", "service_zone"]
    )

    zones["LocationID"] = pd.to_numeric(
        zones["LocationID"], errors="coerce"
    )
    zones = zones.dropna(subset=["LocationID"])
    zones["LocationID"] = zones["LocationID"].astype(int)

    zones["Zone"] = zones["Zone"].fillna("Unknown").astype(str)
    zones["Borough"] = zones["Borough"].fillna("Unknown").astype(str)

    zones["display_name"] = (
        zones["Zone"] + " — " + zones["Borough"]
        + " (ID: " + zones["LocationID"].astype(str) + ")"
    )

    return zones


@st.cache_resource
def load_models():
    duration_model = xgb.Booster()
    duration_model.load_model(
        "xgboost_trip_models/duration_min_xgboost.json"
    )

    amount_model = xgb.Booster()
    amount_model.load_model(
        "xgboost_trip_models/total_amount_xgboost.json"
    )

    return duration_model, amount_model


try:
    zones = load_zones()
    duration_model, amount_model = load_models()
except Exception as e:
    st.error(f"Could not load data or models: {e}")
    st.stop()

if zones.empty:
    st.error("No taxi zones were found in taxi_zone_lookup.csv.")
    st.stop()


# ==================== HEADER ====================
st.markdown("""
<div class="hero">
    <div class="eyebrow">Urban Traffic & Mobility Analytics</div>
    <h1>NYC Taxi Trip Predictor</h1>
    <p>
        Estimate trip duration and total amount using machine learning
        models trained on historical NYC taxi trip data.
    </p>
</div>
""", unsafe_allow_html=True)

# Compact project overview
overview1, overview2, overview3 = st.columns(3)

with overview1:
    st.metric("Prediction Models", "XGBoost", help="Gradient-boosted tree models")

with overview2:
    st.metric("Prediction Targets", "2", help="Trip duration and total amount")

with overview3:
    st.metric("Input Locations", f"{zones['LocationID'].nunique():,}")


# ==================== TRIP INPUTS ====================
st.markdown(
    '<div class="section-heading">Plan your trip</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="panel">', unsafe_allow_html=True)

zone_options = zones["display_name"].tolist()


def default_zone_index(location_id):
    matches = zones.index[zones["LocationID"] == location_id].tolist()

    if matches:
        selected_name = zones.loc[matches[0], "display_name"]
        return zone_options.index(selected_name)

    return 0


with st.form("trip_prediction_form"):
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**Origin & Distance**")

        pickup_name = st.selectbox(
            "Pickup Zone",
            zone_options,
            index=default_zone_index(161)
        )

        trip_distance = st.number_input(
            "Trip Distance (miles)",
            min_value=0.1,
            max_value=100.0,
            value=3.1,
            step=0.1,
            help="Enter the estimated trip distance in miles."
        )

        trip_date = st.date_input(
            "Pickup Date",
            value=date.today(),
            min_value=date(2022, 1, 1),
            max_value=date(2026, 12, 31)
        )

    with right:
        st.markdown("**Destination & Schedule**")

        dropoff_name = st.selectbox(
            "Drop-off Zone",
            zone_options,
            index=default_zone_index(237)
        )

        passenger_count = st.selectbox(
            "Passenger Count",
            options=[1, 2, 3, 4, 5, 6],
            index=0
        )

        trip_time = st.time_input(
            "Pickup Time",
            value=time(14, 0)
        )

    submitted = st.form_submit_button(
        "Predict Trip",
        type="primary",
        use_container_width=True
    )

st.markdown('</div>', unsafe_allow_html=True)


# ==================== PREDICTION ====================
if submitted:
    try:
        pickup_zone = int(
            zones.loc[
                zones["display_name"] == pickup_name,
                "LocationID"
            ].iloc[0]
        )

        dropoff_zone = int(
            zones.loc[
                zones["display_name"] == dropoff_name,
                "LocationID"
            ].iloc[0]
        )

        # Spark convention: Sunday=1, Saturday=7
        spark_dayofweek = ((trip_date.weekday() + 1) % 7) + 1
        is_weekend = int(spark_dayofweek in [1, 7])

        # Must match the model's training feature names and order
        feature_order = [
            "pu_zone",
            "do_zone",
            "passenger_count",
            "trip_distance",
            "pickup_hour",
            "day_of_week",
            "month",
            "year",
            "is_weekend"
        ]

        features = pd.DataFrame([{
            "pu_zone": pickup_zone,
            "do_zone": dropoff_zone,
            "passenger_count": passenger_count,
            "trip_distance": float(trip_distance),
            "pickup_hour": trip_time.hour,
            "day_of_week": spark_dayofweek,
            "month": trip_date.month,
            "year": trip_date.year,
            "is_weekend": is_weekend
        }])[feature_order]

        dtest = xgb.DMatrix(
            features,
            feature_names=feature_order
        )

        duration = max(
            0.0, float(duration_model.predict(dtest)[0])
        )
        amount = max(
            0.0, float(amount_model.predict(dtest)[0])
        )

        st.markdown(
            '<div class="section-heading">Prediction Results</div>',
            unsafe_allow_html=True
        )

        result1, result2 = st.columns(2, gap="large")

        with result1:
            st.markdown(f"""
            <div class="panel">
                <div class="result-label">Estimated Trip Duration</div>
                <div class="result-value">{duration:.1f} min</div>
                <div class="result-unit">Approximately {duration / 60:.2f} hours</div>
            </div>
            """, unsafe_allow_html=True)

        with result2:
            st.markdown(f"""
            <div class="panel">
                <div class="result-label">Estimated Total Amount</div>
                <div class="result-value">${amount:,.2f}</div>
                <div class="result-unit">Estimated amount in USD</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(
            '<div class="section-heading">Trip Summary</div>',
            unsafe_allow_html=True
        )

        summary1, summary2, summary3 = st.columns(3)

        with summary1:
            st.markdown(f"""
            <div class="route-card">
                <div class="route-label">PICKUP</div>
                <div class="route-value">{pickup_name}</div>
            </div>
            """, unsafe_allow_html=True)

        with summary2:
            st.markdown(f"""
            <div class="route-card">
                <div class="route-label">DROP-OFF</div>
                <div class="route-value">{dropoff_name}</div>
            </div>
            """, unsafe_allow_html=True)

        with summary3:
            st.markdown(f"""
            <div class="route-card">
                <div class="route-label">TRIP DETAILS</div>
                <div class="route-value">{trip_distance:.1f} miles</div>
                <div class="route-label">{passenger_count} passenger(s)</div>
            </div>
            """, unsafe_allow_html=True)

        st.caption(
            f"Pickup: {trip_date.strftime('%b %d, %Y')} at "
            f"{trip_time.strftime('%H:%M')} · "
            f"Pickup zone ID: {pickup_zone} · "
            f"Drop-off zone ID: {dropoff_zone}"
        )

        with st.expander("View model input features"):
            st.dataframe(features, use_container_width=True)

        st.info(
            "These predictions are estimates based on historical data. "
            "Actual travel time and total amount may differ because of "
            "traffic, route choice, tolls, and other conditions."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    NYC Traffic & Mobility Analytics · Machine Learning Prediction
</div>
""", unsafe_allow_html=True)
