import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# Title
# =========================
st.title("📊 KNN Regression Analysis App")

# =========================
# File Upload
# =========================
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # =========================
    # Handle Missing Values
    # =========================
    if "income" in df.columns:
        df["income"] = df["income"].fillna(df["income"].mean())
    if "loan_amount" in df.columns:
        df["loan_amount"] = df["loan_amount"].fillna(df["loan_amount"].median())
    if "credit_score" in df.columns:
        df["credit_score"] = df["credit_score"].fillna(df["credit_score"].mean())
    if "annual_spend" in df.columns:
        df["annual_spend"] = df["annual_spend"].fillna(df["annual_spend"].median())

    # Drop date column
    if "date" in df.columns:
        df.drop("date", axis=1, inplace=True)

    # =========================
    # Encoding
    # =========================
    categorical_cols = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=categorical_cols, dtype=int)

    # =========================
    # Scaling
    # =========================
    num_cols = df.select_dtypes(include=np.number).columns
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # =========================
    # Split Data
    # =========================
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # K Selection
    # =========================
    k = st.slider("Select K value", 1, 20, 5)

    # =========================
    # Train Model
    # =========================
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predictions
    y_pred = knn.predict(X_test)

    # =========================
    # Metrics
    # =========================
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📊 Model Performance")
    st.write(f"**MSE:** {mse}")
    st.write(f"**R2 Score (Accuracy):** {r2}")

    # =========================
    # Elbow Plot
    # =========================
    st.subheader("📈 Elbow Plot")

    mse_values = []
    k_values = range(1, 21)

    for k_val in k_values:
        model = KNeighborsRegressor(n_neighbors=k_val)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse_values.append(mean_squared_error(y_test, pred))

    fig, ax = plt.subplots()
    ax.plot(k_values, mse_values, marker='o')
    ax.set_xlabel("K Value")
    ax.set_ylabel("MSE")
    ax.set_title("Elbow Plot")

    st.pyplot(fig)

    # =========================
    # Prediction Section
    # =========================
    st.subheader("🔮 Make Prediction")

    input_data = []
    for col in X.columns:
        val = st.number_input(f"Enter {col}", value=0.0)
        input_data.append(val)

    if st.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        prediction = knn.predict(input_array)
        st.success(f"Predicted Value: {prediction[0]}")