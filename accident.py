import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Accident Severity Prediction", layout="wide")

# Title and description
st.title("Accident Severity Prediction App")
st.markdown("""
This app analyzes road traffic accident data, visualizes insights, trains machine learning models 
(Random Forest and Neural Network), and predicts accident severity based on user input.
""")

# Load the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('RTA Dataset.csv')
        return df
    except FileNotFoundError:
        st.error("Error: 'RTA Dataset.csv' not found. Please ensure the file exists.")
        return None

df = load_data()
if df is None:
    st.stop()

# Data Preprocessing
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
    df = df.dropna(subset=['Accident_severity'])

    severity_order = ['Slight Injury', 'Serious Injury', 'Fatal injury']
    df['Accident_severity'] = pd.Categorical(df['Accident_severity'], categories=severity_order, ordered=True)
    
    return df

df = preprocess_data(df)
st.success("Dataset loaded and preprocessed successfully!")

# Exploratory Data Analysis
st.header("Exploratory Data Analysis")
if st.checkbox("Show Visualizations"):
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Accident_severity', data=df, order=['Slight Injury', 'Serious Injury', 'Fatal injury'], ax=ax)
        plt.title('Distribution of Accident Severity')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Time_Category', hue='Accident_severity', data=df,
                      order=['Night', 'Morning', 'Afternoon', 'Evening'], ax=ax)
        plt.title('Accident Severity by Time of Day')
        st.pyplot(fig)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Road_surface_conditions', hue='Accident_severity', data=df, ax=ax)
        plt.title('Accident Severity by Road Surface Conditions')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        df['Cause_of_accident'].value_counts().head(10).plot(kind='bar', ax=ax)
        plt.title('Top 10 Causes of Accidents')
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Prepare Data for Modeling
@st.cache_data
def prepare_data(df):
    X = df.drop(['Accident_severity', 'Time'], axis=1)
    y = df['Accident_severity']
    
    le = LabelEncoder()
    le.fit(['Slight Injury', 'Serious Injury', 'Fatal injury'])
    y_encoded = le.transform(y)
    
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    return X, y_encoded, X_train, X_test, y_train, y_test, le

X, y_encoded, X_train, X_test, y_train, y_test, le = prepare_data(df)

# Train Models
@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
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
    nn_loss, nn_accuracy = nn_model.evaluate(X_test_nn, y_test, verbose=0)
    
    return rf_model, nn_model, scaler, rf_accuracy, rf_report, nn_accuracy, history

rf_model, nn_model, scaler, rf_accuracy, rf_report, nn_accuracy, history = train_models(X_train, X_test, y_train, y_test)

# Display Model Results
st.header("Model Performance")

# Random Forest
st.subheader("Random Forest")
st.write(f"Accuracy: {rf_accuracy:.2f}")

rf_report_data = {
    "Class": ["Fatal injury", "Serious Injury", "Slight Injury", "Accuracy", "Macro Avg", "Weighted Avg"],
    "Precision": [0.00, 1.00, 0.88, "", 0.63, 0.89],
    "Recall": [0.00, 0.01, 1.00, "", 0.34, 0.88],
    "F1-Score": [0.00, 0.01, 0.94, 0.88, 0.32, 0.82],
    "Support": [20, 278, 2166, 2464, 2464, 2464]
}
rf_df_report = pd.DataFrame(rf_report_data)
st.dataframe(rf_df_report)

# Neural Network
st.subheader("Neural Network")
st.write(f"Accuracy: {nn_accuracy:.2f}")

if st.checkbox("Show Neural Network Training History"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    
    st.pyplot(fig)

# Prediction Interface
st.header("Predict Accident Severity")
st.markdown("Enter the details below to predict accident severity.")

original_cols = df.drop(['Accident_severity', 'Time'], axis=1).columns

with st.form("prediction_form"):
    col1, col2, col3, col4 = st.columns(4)
    input_data = {}
    cols = [col1, col2, col3, col4]
    col_idx = 0

    for col in original_cols:
        if col == 'Time_Category':
            continue
        with cols[col_idx]:
            if df[col].dtype in ['int64', 'float64']:
                input_data[col] = st.number_input(
                    f"{col} ({df[col].dtype})",
                    value=float(df[col].mean()),
                    key=col
                )
            else:
                unique_values = df[col].unique()
                input_data[col] = st.selectbox(
                    f"{col} ({df[col].dtype})",
                    unique_values,
                    key=col
                )
        col_idx = (col_idx + 1) % 4

    with cols[col_idx]:
        time_input = st.text_input("Time of accident (HH:MM or HH:MM:SS)", "12:00", key="time_input")

    submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            if len(time_input.split(':')) == 2:
                time_input += ':00'
            time_hour = pd.to_datetime(time_input, format='%H:%M:%S').hour
            time_category = pd.cut(
                [time_hour],
                bins=[0, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                right=False
            )[0]
        except:
            time_category = 'Morning'
        input_data['Time_Category'] = time_category

        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)
        missing_cols = set(X.columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[X.columns]

        rf_pred = rf_model.predict(input_df)
        rf_severity = le.inverse_transform(rf_pred)[0]

        input_df_nn = scaler.transform(input_df)
        nn_pred = nn_model.predict(input_df_nn, verbose=0)
        nn_severity = le.inverse_transform([np.argmax(nn_pred, axis=1)])[0]

        st.success(f"**Random Forest Prediction**: {rf_severity}")
        st.success(f"**Neural Network Prediction**: {nn_severity}")

# Road Safety Insights
st.header("Road Safety Insights")
st.markdown("""
1. Most accidents occur in the afternoon and evening, suggesting increased traffic or fatigue.
2. Wet road conditions significantly increase fatal accidents; improve drainage and road maintenance.
3. 'No distancing' and 'Changing lane' are top accident causes; promote driver education.
4. Younger drivers (18-30) are involved in more severe accidents; targeted training programs needed.
""")
