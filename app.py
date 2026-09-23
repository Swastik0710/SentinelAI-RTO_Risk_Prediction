import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="SentinelAI - RTO Risk Prediction",
    page_icon="🚚",
    layout="wide"
)

model = joblib.load("rto_prediction_model.pkl")

feature_names = [
    "order_id","customer_id","seller_id","order_date","order_value",
    "quantity","product_category","discount_pct","payment_method",
    "customer_state","delivery_zone","customer_order_count",
    "customer_previous_rto_count","customer_previous_cancel_count",
    "previous_successful_orders","customer_previous_rto_rate",
    "days_since_last_order","seller_previous_orders",
    "seller_previous_rto_count","seller_previous_rto_rate",
    "orders_last_24h","orders_last_7d","device_order_count_7d",
    "multiple_accounts_device","address_order_count",
    "distance_seller_customer_km","estimated_delivery_days",
    "is_weekend","is_holiday","seller_rating",
    "customer_account_age_days","cod_amount",
    "address_change_count_30d",
    "failed_delivery_attempts_previous","night_order"
]

st.title("🚚 SentinelAI - RTO Risk Prediction")
st.markdown(
    "Predict the **Return-To-Origin (RTO)** risk of an e-commerce order using a trained **Random Forest Machine Learning model**."
)

st.divider()

left, right = st.columns(2)

with left:
    order_value = st.number_input(
        "Order Value (₹)",
        min_value=0.0,
        value=None,
        placeholder="Enter order value"
    )

    quantity = st.number_input(
        "Quantity",
        min_value=1,
        value=None,
        placeholder="Enter quantity"
    )

    discount_pct = st.slider("Discount %", 0, 100, 0)

    payment_method = st.selectbox(
        "Payment Method",
        ["Select","UPI","COD","Credit Card","Debit Card","Wallet"],
        index=0
    )

    customer_state = st.number_input(
        "Customer State (Encoded)",
        min_value=0,
        value=None,
        placeholder="Enter state code"
    )

    delivery_zone = st.number_input(
        "Delivery Zone",
        min_value=0,
        value=None,
        placeholder="Enter delivery zone"
    )

    customer_order_count = st.number_input(
        "Customer Order Count",
        min_value=0,
        value=None,
        placeholder="Enter order count"
    )

    customer_previous_rto_rate = st.number_input(
        "Previous RTO Rate",
        min_value=0.0,
        max_value=1.0,
        value=None,
        placeholder="0.00"
    )

with right:
    seller_previous_rto_rate = st.number_input(
        "Seller RTO Rate",
        min_value=0.0,
        max_value=1.0,
        value=None,
        placeholder="0.00"
    )

    distance = st.number_input(
        "Distance (km)",
        min_value=0.0,
        value=None,
        placeholder="Enter distance"
    )

    delivery_days = st.number_input(
        "Estimated Delivery Days",
        min_value=1,
        value=None,
        placeholder="Enter delivery days"
    )

    seller_rating = st.slider("Seller Rating", 1.0, 5.0, 3.0)

    customer_account_age = st.number_input(
        "Customer Account Age (Days)",
        min_value=0,
        value=None,
        placeholder="Enter account age"
    )

    cod_amount = st.number_input(
        "COD Amount (₹)",
        min_value=0.0,
        value=None,
        placeholder="Enter COD amount"
    )

    is_weekend = st.selectbox(
        "Weekend Order",
        ["Select","No","Yes"],
        index=0
    )

    night_order = st.selectbox(
        "Night Order",
        ["Select","No","Yes"],
        index=0
    )

st.divider()

if st.button("🔍 Predict RTO Risk", use_container_width=True):

    required_numbers = [
        order_value, quantity, customer_state, delivery_zone,
        customer_order_count, customer_previous_rto_rate,
        seller_previous_rto_rate, distance,
        delivery_days, customer_account_age, cod_amount
    ]

    if (
        any(v is None for v in required_numbers)
        or payment_method == "Select"
        or is_weekend == "Select"
        or night_order == "Select"
    ):
        st.warning("⚠ Please complete all fields before prediction.")
        st.stop()

    payment_encoding = {
        "UPI":0,
        "COD":1,
        "Credit Card":2,
        "Debit Card":3,
        "Wallet":4
    }

    weekend = 1 if is_weekend == "Yes" else 0
    night = 1 if night_order == "Yes" else 0

    sample = pd.DataFrame([{
        "order_id":0,
        "customer_id":0,
        "seller_id":0,
        "order_date":0,
        "order_value":order_value,
        "quantity":quantity,
        "product_category":0,
        "discount_pct":discount_pct,
        "payment_method":payment_encoding[payment_method],
        "customer_state":customer_state,
        "delivery_zone":delivery_zone,
        "customer_order_count":customer_order_count,
        "customer_previous_rto_count":1,
        "customer_previous_cancel_count":0,
        "previous_successful_orders":customer_order_count,
        "customer_previous_rto_rate":customer_previous_rto_rate,
        "days_since_last_order":30,
        "seller_previous_orders":100,
        "seller_previous_rto_count":5,
        "seller_previous_rto_rate":seller_previous_rto_rate,
        "orders_last_24h":1,
        "orders_last_7d":5,
        "device_order_count_7d":1,
        "multiple_accounts_device":0,
        "address_order_count":1,
        "distance_seller_customer_km":distance,
        "estimated_delivery_days":delivery_days,
        "is_weekend":weekend,
        "is_holiday":0,
        "seller_rating":seller_rating,
        "customer_account_age_days":customer_account_age,
        "cod_amount":cod_amount,
        "address_change_count_30d":0,
        "failed_delivery_attempts_previous":0,
        "night_order":night
    }])

    sample = sample[feature_names]

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    st.subheader("📊 Prediction Result")

    c1, c2 = st.columns(2)

    with c1:
        if prediction == 1:
            st.error("### 🔴 HIGH RTO RISK")
        else:
            st.success("### 🟢 LOW RTO RISK")

    with c2:
        st.metric("RTO Probability", f"{probability*100:.2f}%")

    st.progress(float(probability))

    st.divider()

    st.subheader("⭐ Top 10 Important Features")

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.bar_chart(importance.head(10).set_index("Feature"))

    st.dataframe(
        importance.head(10),
        use_container_width=True,
        hide_index=True
    )

st.divider()
st.caption("Developed using Python • Pandas • Scikit-learn • Streamlit")
