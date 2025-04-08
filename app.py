import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")

st.title("💳 Credit Card Fraud Detection System")
st.markdown("Predict if a transaction is **Legit** or **Fraudulent** using Logistic Regression.")

# --- Load and train model ---
@st.cache_data
def train_model():
    df = pd.read_csv("creditcard.csv")

    # Undersample to balance classes
    legit = df[df.Class == 0]
    fraud = df[df.Class == 1]
    legit_sample = legit.sample(n=len(fraud), random_state=42)
    balanced_data = pd.concat([legit_sample, fraud], axis=0)

    X = balanced_data.drop("Class", axis=1)
    y = balanced_data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, X.columns.tolist(), accuracy

model, feature_names, test_acc = train_model()

# --- Sidebar Mode Selection ---
st.sidebar.header("⚙️ Choose Prediction Mode")
mode = st.sidebar.radio("Select input method:", ["📂 Upload CSV File", "📝 Manual Input"])

# --- CSV Upload ---
if mode == "📂 Upload CSV File":
    st.subheader("📂 Upload a CSV File")
    uploaded_file = st.file_uploader("Upload file containing transaction data", type=["csv"])

    if uploaded_file is not None:
        try:
            input_data = pd.read_csv(uploaded_file)

            # Check required columns
            if all(col in input_data.columns for col in feature_names):
                preds = model.predict(input_data[feature_names])
                input_data["Prediction"] = preds
                input_data["Prediction"] = input_data["Prediction"].map({0: "Legit ✅", 1: "Fraud 🚨"})

                st.success("Prediction completed!")
                st.dataframe(input_data)

                csv = input_data.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Results CSV", csv, "fraud_predictions.csv", "text/csv")

            else:
                st.error("❌ Uploaded CSV must contain all required feature columns.")

        except Exception as e:
            st.error(f"❌ Error reading the CSV file: {e}")

# --- Manual Input ---
elif mode == "📝 Manual Input":
    st.subheader("📝 Enter Transaction Details")

    col_layout = st.columns(4)
    user_input = {}

    for i, feature in enumerate(feature_names):
        with col_layout[i % 4]:
            user_input[feature] = st.number_input(f"{feature}", value=0.0, format="%.4f")

    if st.button("🔍 Predict"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        confidence = model.predict_proba(input_df)[0][prediction]

        if prediction == 1:
            st.error(f"🚨 Fraudulent Transaction Detected! (Confidence: {confidence:.2f})")
        else:
            st.success(f"✅ Legit Transaction Detected. (Confidence: {confidence:.2f})")

# --- Footer ---
st.markdown("---")
st.caption(f"🔎 Logistic Regression Model Accuracy on Test Data: **{test_acc * 100:.2f}%**")
