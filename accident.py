import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import streamlit as st

# Set page config
st.set_page_config(page_title="Accident Severity Prediction", layout="wide")

# Title
st.title("Accident Severity Prediction App")
st.markdown("""
This app analyzes road traffic accident data, visualizes insights, trains machine learning models 
(Random Forest and Neural Network), and predicts accident severity based on user input.
""")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("RTA Dataset.csv")
    return df

df = load_data()

# Preprocessing
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
            if pd.isna(time_str): return pd.NaT
            if isinstance(time_str, str):
                if len(time_str.split(':')) == 2: time_str += ':00'
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
    df.dropna(subset=['Accident_severity'], inplace=True)
    df['Accident_severity'] = pd.Categorical(df['Accident_severity'], 
                                              categories=['Slight Injury', 'Serious Injury', 'Fatal injury'], ordered=True)
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

# Train Models
@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_report = classification_report(y_test, y_pred_rf, target_names=le.classes_)

    scaler = StandardScaler()
    X_train_nn = scaler.fit_transform(X_train)
    X_test_nn = scaler.transform(X_test)

    nn_model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(len(le.classes_), activation='softmax')
    ])
    nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = nn_model.fit(X_train_nn, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
    nn_accuracy = nn_model.evaluate(X_test_nn, y_test, verbose=0)[1]

    return rf_model, nn_model, scaler, rf_accuracy, rf_report, nn_accuracy, history

rf_model, nn_model, scaler, rf_accuracy, rf_report, nn_accuracy, history = train_models(X_train, X_test, y_train, y_test)

# Display Results
st.header("Model Performance")
st.subheader("Random Forest")
st.write(f"Accuracy: {rf_accuracy:.2f}")

def classification_report_to_df(report_str):
    lines = report_str.split('\n')
    data = []
    for line in lines[2:-5]:
        row = line.split()
        if len(row) > 0:
            label = row[0]
            if label not in ['accuracy', 'macro', 'weighted']:
                data.append([label] + list(map(float, row[1:])))
    df_report = pd.DataFrame(data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
    return df_report

df_rf_report = classification_report_to_df(rf_report)
st.dataframe(df_rf_report.style.set_caption("Random Forest Classification Report").format(precision=2))

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, rf_model.predict(X_test))
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# Prediction
st.header("Predict Accident Severity")
input_data = {}
original_cols = df.drop(['Accident_severity', 'Time'], axis=1).columns

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    for i, col_name in enumerate(original_cols):
        if col_name == 'Time_Category':
            continue
        with col1 if i % 2 == 0 else col2:
            if df[col_name].dtype in ['int64', 'float64']:
                input_data[col_name] = st.number_input(f"{col_name}", value=float(df[col_name].mean()))
            else:
                input_data[col_name] = st.selectbox(f"{col_name}", df[col_name].unique())

    time_input = st.text_input("Time of accident (HH:MM or HH:MM:SS)", "12:00")
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        if len(time_input.split(':')) == 2:
            time_input += ':00'
        hour = pd.to_datetime(time_input, format='%H:%M:%S').hour
        category = pd.cut([hour], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])[0]
    except:
        category = 'Morning'
    input_data['Time_Category'] = category

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    missing_cols = set(X.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[X.columns]

    rf_pred = rf_model.predict(input_df)
    rf_severity = le.inverse_transform(rf_pred)[0]

    nn_input = scaler.transform(input_df)
    nn_pred = nn_model.predict(nn_input, verbose=0)
    nn_severity = le.inverse_transform([np.argmax(nn_pred, axis=1)])[0]

    st.success(f"Random Forest Prediction: {rf_severity}")
    st.success(f"Neural Network Prediction: {nn_severity}")

    st.subheader("Prediction Visual")
    pred_values = [0] * len(le.classes_)
    pred_values[rf_pred[0]] = 1
    fig_pred, ax_pred = plt.subplots()
    sns.barplot(x=le.classes_, y=pred_values, palette='Set2', ax=ax_pred)
    ax_pred.set_ylabel("Predicted Class Indicator")
    st.pyplot(fig_pred)

# Safety Tips
st.header("Road Safety Insights")
st.markdown("""
- Afternoon and evening have high accident rates.
- Wet roads increase fatal accidents.
- Driver misjudgment and no distancing are top causes.
- Younger drivers (18-30) involved in severe cases.
""")
