"""
Hotel Booking Cancellation Predictor
A Streamlit decision-support app for hotel revenue managers.
Run locally:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent

st.set_page_config(page_title="Hotel Cancellation Predictor", page_icon="🏨", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(BASE / "hotel_cancellation_model.pkl")


@st.cache_data
def load_metrics():
    try:
        return json.loads((BASE / "model_metrics.json").read_text())
    except Exception:
        return None


model = load_model()
metrics = load_metrics()

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

# ---------------- Header ----------------
st.title("🏨 Hotel Booking Cancellation Predictor")
st.write(
    "Enter the details of a booking to estimate the **chance that the guest will cancel**, "
    "and get a **recommended action** for the hotel's revenue team."
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About this app")
    st.write(
        "Hotels lose revenue when bookings are cancelled and rooms stay empty. "
        "This app uses a machine learning model trained on 25,000+ past hotel bookings "
        "to flag risky bookings early."
    )
    if metrics:
        st.subheader("Model performance (test data)")
        st.write(f"**Model:** {metrics['model']}")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
        c2.metric("ROC-AUC", f"{metrics['ROC-AUC']:.2f}")
        c1.metric("Precision", f"{metrics['Precision']:.1%}")
        c2.metric("Recall", f"{metrics['Recall']:.1%}")
    st.caption("Data: Hotel Reservations Dataset (Kaggle). For academic use.")

# ---------------- Inputs ----------------
st.subheader("Booking details")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Guests and stay**")
    adults = st.number_input("Adults", 0, 4, 2)
    children = st.number_input("Children", 0, 3, 0)
    weekend_nights = st.number_input("Weekend nights", 0, 7, 1)
    week_nights = st.number_input("Week nights", 0, 17, 2)

with col2:
    st.markdown("**Booking**")
    lead_time = st.slider("Lead time (days between booking and arrival)", 0, 450, 60)
    month = st.selectbox("Arrival month", MONTHS, index=7)
    segment = st.selectbox("Booking channel (market segment)",
                           ["Online", "Offline", "Corporate", "Aviation", "Complementary"])
    price = st.number_input("Average price per room per night (€)", 0.0, 500.0, 100.0, step=5.0)

with col3:
    st.markdown("**Preferences and history**")
    meal = st.selectbox("Meal plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
    room = st.selectbox("Room type", [f"Room_Type {i}" for i in range(1, 8)])
    special = st.slider("Number of special requests", 0, 5, 0)
    parking = st.checkbox("Needs car parking")
    repeated = st.checkbox("Repeat guest (has stayed before)")
    prev_cancel = st.number_input("Previous cancellations by this guest", 0, 15, 0)
    prev_ok = st.number_input("Previous completed stays", 0, 60, 0)

if adults + children == 0:
    st.warning("Please enter at least one guest.")
    st.stop()
if weekend_nights + week_nights == 0:
    st.warning("Please enter at least one night.")
    st.stop()

booking = pd.DataFrame([{
    "no_of_adults": adults,
    "no_of_children": children,
    "no_of_weekend_nights": weekend_nights,
    "no_of_week_nights": week_nights,
    "required_car_parking_space": int(parking),
    "lead_time": lead_time,
    "arrival_month": MONTHS.index(month) + 1,
    "repeated_guest": int(repeated),
    "no_of_previous_cancellations": prev_cancel,
    "no_of_previous_bookings_not_canceled": prev_ok,
    "avg_price_per_room": price,
    "no_of_special_requests": special,
    "type_of_meal_plan": meal,
    "room_type_reserved": room,
    "market_segment_type": segment,
}])

st.divider()

# ---------------- Prediction ----------------
if st.button("Predict cancellation risk", type="primary", use_container_width=True):
    p = float(model.predict_proba(booking)[0, 1])
    nights = weekend_nights + week_nights
    revenue = price * nights

    if p >= 0.70:
        level, color = "HIGH", "red"
    elif p >= 0.40:
        level, color = "MEDIUM", "orange"
    else:
        level, color = "LOW", "green"

    r1, r2, r3 = st.columns(3)
    r1.metric("Cancellation probability", f"{p:.0%}")
    r2.metric("Risk level", level)
    r3.metric("Revenue at risk (expected)", f"€{p * revenue:,.0f}", help=f"{p:.0%} × €{revenue:,.0f} booking value")
    st.progress(p)

    st.subheader(f":{color}[{level} RISK]: recommended action")
    if level == "HIGH":
        st.error(
            "- Ask for a **non-refundable deposit or prepayment** before confirming.\n"
            "- Send **reminder emails/SMS** before arrival and ask the guest to reconfirm.\n"
            "- Count this booking when planning **controlled overbooking** for these dates.\n"
            "- Offer a small incentive (e.g., free breakfast) if the guest reconfirms."
        )
    elif level == "MEDIUM":
        st.warning(
            "- Send a **confirmation reminder** about 7 days before arrival.\n"
            "- Offer a **perk or upgrade** to increase commitment.\n"
            "- Keep an eye on this booking in the weekly cancellation report."
        )
    else:
        st.success(
            "- **Confirm normally.** This guest is likely to arrive.\n"
            "- Focus on guest experience and **upselling** (spa, dining, late checkout)."
        )

    # Why: simple, rule-based explanation using patterns found in the EDA
    reasons = []
    if lead_time > 150:
        reasons.append(f"🔺 Booked **{lead_time} days in advance**: long lead times are linked to many more cancellations.")
    elif lead_time <= 30:
        reasons.append("🔻 Short lead time (≤30 days): last-minute bookings rarely cancel.")
    if special == 0:
        reasons.append("🔺 **No special requests**: guests without requests cancel more often.")
    else:
        reasons.append(f"🔻 {special} special request(s): a sign the guest is committed.")
    if segment == "Online":
        reasons.append("🔺 **Online booking**: the channel with the highest cancellation rate.")
    elif segment in ("Corporate", "Complementary"):
        reasons.append(f"🔻 {segment} booking: this channel rarely cancels.")
    if repeated:
        reasons.append("🔻 Repeat guest: loyal guests almost never cancel.")
    if prev_cancel > 0:
        reasons.append(f"🔺 Guest has **{prev_cancel} previous cancellation(s)**.")
    if month in ("July", "August"):
        reasons.append(f"🔺 Arrival in **{month}**: the peak season for cancellations.")
    elif month in ("January", "December"):
        reasons.append(f"🔻 Arrival in {month}: cancellations are lowest in this season.")
    if price > 150:
        reasons.append("🔺 High room price: expensive bookings are cancelled slightly more often.")

    st.subheader("Why this result?")
    for r in reasons:
        st.markdown(r)
    st.caption("🔺 increases risk · 🔻 reduces risk. Explanation based on patterns found in the historical data.")
else:
    st.info("Fill in the booking details above and click **Predict cancellation risk**.")
