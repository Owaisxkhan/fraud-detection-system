import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# streamlit cloud fix applied


from prediction_helper import predict_fraud

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Transaction Fraud Detection",
    page_icon="🚨",
    layout="centered"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.card {
    background-color: #1e1e1e;
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
}
.title {
    font-size: 36px;
    font-weight: 700;
}
.subtitle {
    color: #b0b0b0;
    margin-bottom: 20px;
}
.result-card {
    padding: 20px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown('<div class="title">🚨 Transaction Fraud Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter transaction details to check fraud risk</div>', unsafe_allow_html=True)

# ---------------- Input Card ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)

with row1[0]:
    step = st.number_input("⏱ Time Step (Hour)", min_value=0, value=10)

with row1[1]:
    tx_type = st.selectbox(
        "💳 Transaction Type",
        ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    )

with row1[2]:
    amount = st.number_input("💰 Transaction Amount", min_value=0.0, value=50000.0)

with row2[0]:
    oldbalanceOrg = st.number_input("👤 Sender Old Balance", min_value=0.0, value=60000.0)

with row2[1]:
    newbalanceOrig = st.number_input("👤 Sender New Balance", min_value=0.0, value=10000.0)

with row2[2]:
    oldbalanceDest = st.number_input("🏦 Receiver Old Balance", min_value=0.0, value=0.0)

with row3[0]:
    newbalanceDest = st.number_input("🏦 Receiver New Balance", min_value=0.0, value=50000.0)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Predict Button ----------------
st.markdown("<br>", unsafe_allow_html=True)
check = st.button("🔍 Check Fraud", use_container_width=True)

# ---------------- Prediction ----------------
if check:
    result = predict_fraud(
        step,
        tx_type,
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    if result["is_fraud"] == 1:
        st.markdown(
            f"""
            <div class="result-card" style="background-color:#3a1f1f;color:#ff6b6b;">
            🚨 Fraud Detected <br><br>
            Fraud Probability: {result['fraud_probability']:.2%}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="result-card" style="background-color:#1f3a2d;color:#5cff9d;">
            ✅ Transaction is Safe <br><br>
            Fraud Probability: {result['fraud_probability']:.2%}
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------- Footer ----------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<center style='color:gray;'>Fraud Detection System • Machine Learning Project</center>",
    unsafe_allow_html=True
)
