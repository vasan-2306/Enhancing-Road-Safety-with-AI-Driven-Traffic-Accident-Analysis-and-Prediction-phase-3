# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import plotly.figure_factory as ff

# Page Config
st.set_page_config(page_title="Accident Severity Predictor", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #111;
            color: #fff;
        }
        .stButton > button {
            background-color: #222;
            color: #fff;
        }
    </style>
""", unsafe_allow_html=True)

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("RTA Dataset.csv")
    return df

df = load_data()

# Preprocess
@st.cache_data
def preprocess_data(df):
    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in numerical_cols:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    def parse_time(time_str):
        try:
            if pd.isna(time_str):
                return pd.NaT
            if isinstance(time_str, str):
                if len(time_str.split(':')) == 2:
                    time_str += ':00'
                return pd.to_datetime(time_str, format='%H:%M:%S', errors='coerce').time()
            return time_str
        except:
            return pd.NaT

    df['Time'] = df['Time'].apply(parse_time)
    df['Time_Category'] = pd.cut(
        pd.to_datetime(df['Time'].astype(str), format='%H:%M:%S', errors='coerce').dt.hour,
        bins=[0, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening'],
        right=False
    )
    df['Time_Category'].fillna('Morning', inplace=True)

    severity_order = ['Slight Injury', 'Serious Injury', 'Fatal injury']
    df['Accident_severity'] = pd.Categorical(df['Accident_severity'], categories=severity_order, ordered=True)
    df = df.dropna(subset=['Accident_severity'])
    return df

df = preprocess_data(df)

# Prepare Data
@st.cache_data
def prepare_data(df):
    X = df.drop(['Accident_severity', 'Time'], axis=1)
    y = df['Accident_severity']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    return X, y_encoded, X_train, X_test, y_train, y_test, le

X, y_encoded, X_train, X_test, y_train, y_test, le = prepare_data(df)

@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return rf, acc, report, cm

rf_model, rf_acc, rf_report, rf_cm = train_models(X_train, X_test, y_train, y_test)

# App Header
st.title("\U0001F6A8 Accident Severity Prediction System")
st.subheader(f"Random Forest Accuracy: {rf_acc:.2f}")

# Plot Classification Report
st.markdown("### Classification Report")
report_df = pd.DataFrame(rf_report).transpose().round(2)
st.dataframe(report_df, use_container_width=True)

# Confusion Matrix
st.markdown("### Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="YlOrBr", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# Landscape Prediction Form
st.markdown("### Predict Accident Severity")
input_data = {}
original_cols = df.drop(['Accident_severity', 'Time'], axis=1).columns

with st.form("prediction_form"):
    cols = st.columns(4)
    for i, col in enumerate(original_cols):
        if col == 'Time_Category':
            continue
        if df[col].dtype in ['int64', 'float64']:
            input_data[col] = cols[i % 4].number_input(f"{col}", value=float(df[col].mean()))
        else:
            unique_vals = df[col].unique()
            input_data[col] = cols[i % 4].selectbox(f"{col}", unique_vals)

    time_input = st.text_input("Time of accident (HH:MM or HH:MM:SS)", "12:00")
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        if len(time_input.split(":")) == 2:
            time_input += ":00"
        hour = pd.to_datetime(time_input, format="%H:%M:%S").hour
        category = pd.cut([hour], bins=[0,6,12,18,24], labels=['Night','Morning','Afternoon','Evening'])[0]
    except:
        category = 'Morning'

    input_data['Time_Category'] = category

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X.columns]

    prediction = rf_model.predict(input_df)
    severity = le.inverse_transform(prediction)[0]
    st.success(f"\U0001F4A1 Predicted Severity: **{severity}**")
