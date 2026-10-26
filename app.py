import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessor
model = joblib.load("random_forest_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
h1 {
    text-align: center;
}

.stApp p {
    text-align: center;
}

div[data-testid="stForm"] {
    max-width: 850px;
    margin-left: auto;
    margin-right: auto;
}

div[data-testid="stFormSubmitButton"] button {
    background-color: #DC2626;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #B91C1C;
    color: white;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# Page heading
st.title("🏦 Bank Marketing Prediction")

st.write(
    "Predict whether a customer is likely to subscribe "
    "to a term deposit using a Random Forest classifier."
)

st.divider()

# Customer form
with st.form("customer_form"):

    # PERSONAL DETAILS
    st.subheader("👤 Personal Details")

    age = st.number_input(
        "Age", min_value=18, max_value=100, value=35
    )

    job = st.selectbox(
        "Job",
        [
            "admin.", "blue-collar", "entrepreneur",
            "housemaid", "management", "retired",
            "self-employed", "services", "student",
            "technician", "unemployed", "unknown"
        ]
    )

    marital = st.selectbox(
        "Marital Status",
        ["married", "single", "divorced"]
    )

    education = st.selectbox(
        "Education",
        ["secondary", "tertiary", "primary", "unknown"]
    )

    st.divider()

    # FINANCIAL DETAILS
    st.subheader("💰 Financial Details")

    balance = st.number_input(
        "Average Yearly Balance", value=1000
    )

    default = st.selectbox(
        "Credit in Default", ["no", "yes"]
    )

    housing = st.selectbox(
        "Housing Loan", ["yes", "no"]
    )

    loan = st.selectbox(
        "Personal Loan", ["no", "yes"]
    )

    st.divider()

    # CAMPAIGN DETAILS
    st.subheader("📞 Campaign Details")

    contact = st.selectbox(
        "Contact Type",
        ["cellular", "telephone", "unknown"]
    )

    day = st.number_input(
        "Contact Day", min_value=1, max_value=31, value=15
    )

    month = st.selectbox(
        "Contact Month",
        [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"
        ]
    )

    duration = st.number_input(
        "Call Duration (seconds)", min_value=0, value=180
    )

    campaign = st.number_input(
        "Contacts During This Campaign",
        min_value=1, value=1
    )

    pdays = st.number_input(
        "Days Since Previous Contact",
        min_value=-1, value=-1
    )

    previous = st.number_input(
        "Previous Contacts", min_value=0, value=0
    )

    poutcome = st.selectbox(
        "Previous Campaign Outcome",
        ["unknown", "failure", "other", "success"]
    )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Subscription",
        use_container_width=True
    )

# Prediction
if submitted:

    customer = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])

    try:
        # Preprocess customer data
        customer_processed = preprocessor.transform(customer)

        # Predict subscription class
        prediction = model.predict(customer_processed)[0]

        # Get probabilities for both classes
        probabilities = model.predict_proba(customer_processed)[0]
        classes = list(model.classes_)

        subscribe_index = classes.index(1)
        not_subscribe_index = classes.index(0)

        # Convert probabilities to percentages
        subscribe_probability = probabilities[subscribe_index] * 100
        not_subscribe_probability = probabilities[not_subscribe_index] * 100

        # Display prediction result
        st.divider()
        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.success(
                "The model predicts that the customer "
                "is likely to subscribe."
            )
        else:
            st.warning(
                "The model predicts that the customer "
                "is unlikely to subscribe."
            )

        # Subscription probability
        st.markdown("### ✅ Probability of Subscribing")
        st.success(f"{subscribe_probability:.2f}%")

        # Non-subscription probability
        st.markdown("### ❌ Probability of Not Subscribing")
        st.error(f"{not_subscribe_probability:.2f}%")

        # Probability bars
        st.markdown("### 📈 Probability Distribution")

        st.progress(
            min(100, max(0, int(round(subscribe_probability)))),
            text=f"Subscribing: {subscribe_probability:.2f}%"
        )

        st.progress(
            min(100, max(0, int(round(not_subscribe_probability)))),
            text=f"Not Subscribing: {not_subscribe_probability:.2f}%"
        )

        st.caption(
            "Probabilities are estimates from the trained model "
            "and do not guarantee actual customer behavior."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
