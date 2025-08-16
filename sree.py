import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# 1. Data loading and cleaning with feature engineering
@st.cache_data
def load_data():
    calories = pd.read_csv('calories.csv')
    exercise = pd.read_csv('exercise.csv')
    exercise_df = exercise.merge(calories, on="User_ID")

    # Remove duplicates and drop User_ID
    exercise_df.drop_duplicates(subset=['User_ID'], keep='last', inplace=True)
    exercise_df.drop(columns="User_ID", inplace=True)

    # Convert Calories to numeric; coerce errors to NaN and drop invalid rows
    exercise_df['Calories'] = pd.to_numeric(exercise_df['Calories'], errors='coerce')
    exercise_df.dropna(subset=['Calories'], inplace=True)

    # Encode Gender: female=0, male=1
    exercise_df = pd.get_dummies(exercise_df, columns=['Gender'], drop_first=True)
    exercise_df.rename(columns={'Gender_male': 'Gender'}, inplace=True)

    # Calculate BMI and round
    exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
    exercise_df["BMI"] = round(exercise_df["BMI"], 2)

    exercise_df.reset_index(drop=True, inplace=True)
    return exercise_df

# Load dataset
exercise_df = load_data()

# 2. Select relevant features and split data
selected_features = ["Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories", "Gender"]

exercise_df = exercise_df[selected_features]
exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Separate features and target before any dummy encoding or processing
X_train = exercise_train_data.drop(columns=["Calories"], axis=1)
y_train = exercise_train_data["Calories"]
X_test = exercise_test_data.drop(columns=["Calories"], axis=1)
y_test = exercise_test_data["Calories"]

# 3. Model initialization and training
linreg = LinearRegression()
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, random_state=6)

linreg.fit(X_train, y_train)
random_reg.fit(X_train, y_train)

# 4. Learning curve plotting function
def plot_learning_curve(model, X, y, X_val, y_val):
    train_errors, val_errors = [], []
    step = max(1, len(X) // 100)
    for m in range(20, len(X), step):
        model.fit(X[:m], y[:m])
        y_train_predict = model.predict(X[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(metrics.mean_squared_error(y[:m], y_train_predict))
        val_errors.append(metrics.mean_squared_error(y_val, y_val_predict))
    plt.figure(figsize=(8, 5))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation")
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("Root Mean Squared Error (RMSE)")
    plt.legend()
    st.pyplot(plt)
    plt.clf()

# 5. Streamlit user interface
st.title("🏋️‍♂️ Calorie Burn Prediction App")

st.sidebar.header("Input Features")
age = st.sidebar.slider("Age", 18, 80, 25)
weight = st.sidebar.number_input("Weight (kg)", 30.0, 150.0, 70.0)
height = st.sidebar.number_input("Height (cm)", 120.0, 220.0, 170.0)
duration = st.sidebar.slider("Exercise Duration (minutes)", 5, 120, 30)
heart_rate = st.sidebar.slider("Heart Rate (bpm)", 60, 200, 100)
body_temp = st.sidebar.number_input("Body Temperature (°C)", 35.0, 42.0, 37.0)
gender = st.sidebar.radio("Gender", ["Female", "Male"])

bmi = round(weight / ((height / 100) ** 2), 2)
gender_encoded = 1 if gender == "Male" else 0

user_input = np.array([[age, bmi, duration, heart_rate, body_temp, gender_encoded]])

# 6. Make predictions
linreg_pred = linreg.predict(user_input)
random_reg_pred = random_reg.predict(user_input)

st.write("## 📊 Prediction Results")
st.write(f"🔹 **Linear Regression Prediction:** {round(linreg_pred[0], 2)} Calories")
st.write(f"🔹 **Random Forest Prediction:** {round(random_reg_pred[0], 2)} Calories")

# 7. Show evaluation metrics for both models
st.write("## 📉 Model Evaluation Metrics")

linreg_test_pred = linreg.predict(X_test)
random_reg_test_pred = random_reg.predict(X_test)

col1, col2 = st.columns(2)

with col1:
    st.write("### 📈 Linear Regression")
    st.write(f"MAE: {round(metrics.mean_absolute_error(y_test, linreg_test_pred), 2)}")
    st.write(f"MSE: {round(metrics.mean_squared_error(y_test, linreg_test_pred), 2)}")
    st.write(f"RMSE: {round(np.sqrt(metrics.mean_squared_error(y_test, linreg_test_pred)), 2)}")

with col2:
    st.write("### 🌳 Random Forest")
    st.write(f"MAE: {round(metrics.mean_absolute_error(y_test, random_reg_test_pred), 2)}")
    st.write(f"MSE: {round(metrics.mean_squared_error(y_test, random_reg_test_pred), 2)}")
    st.write(f"RMSE: {round(np.sqrt(metrics.mean_squared_error(y_test, random_reg_test_pred)), 2)}")

# 8. Plot learning curve for Random Forest
st.write("## 📚 Learning Curve")
plot_learning_curve(random_reg, X_train.values, y_train.values, X_test.values, y_test.values)

st.markdown("---")
st.write("💡 Developed by **Sreejan Narapareddy** | CSE Cybersecurity, CUJ")