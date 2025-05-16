import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("RandomForest.pkl")

# Page title
st.title("Predict Accident Severity")

# Form for user input
with st.form("severity_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        day_of_week = st.selectbox("Day_of_week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        vehicle_driver_relation = st.selectbox("Vehicle_driver_relation", ["Employee", "Owner", "Other"])
        service_year_of_vehicle = st.selectbox("Service_year_of_vehicle", ["0-5yr", "5-10yr", "Above 10yr"])
        road_alignment = st.selectbox("Road_alignment", [
            "Tangent road with flat terrain", 
            "Tangent road with slight slope", 
            "Curved road with flat terrain", 
            "Curved road with slight slope", 
            "Curved road with steep slope"
        ])
        light_conditions = st.selectbox("Light_conditions", ["Daylight", "Darkness - lights lit", "Darkness - no lighting", "Darkness - lights unlit"])
        number_of_casualties = st.number_input("Number_of_casualties", min_value=0.0, step=0.01)

    with col2:
        age_band_of_driver = st.selectbox("Age_band_of_driver", ["Under 18", "18-30", "31-50", "Over 51"])
        driving_experience = st.selectbox("Driving_experience", ["No Licence", "1-2yr", "2-5yr", "5-10yr", "Above 10yr"])
        defect_of_vehicle = st.selectbox("Defect_of_vehicle", ["No defect", "Brake failure", "Tire problem", "Lighting problem", "Other"])
        types_of_junction = st.selectbox("Types_of_Junction", ["No junction", "T-Junction", "Crossing", "Y-Junction", "Other"])
        weather_conditions = st.selectbox("Weather_conditions", ["Normal", "Rainy", "Fog", "Windy", "Other"])
        vehicle_movement = st.selectbox("Vehicle_movement", ["Going straight", "U-turn", "Reversing", "Turning left", "Turning right", "Parked"])

    with col3:
        sex_of_driver = st.selectbox("Sex_of_driver", ["Male", "Female", "Unknown"])
        educational_level = st.selectbox("Educational_level", ["Below high school", "Above high school", "Unknown"])
        area_accident_occured = st.selectbox("Area_accident_occured", ["Residential areas", "Office areas", "Industrial areas", "School areas"])
        lanes_or_medians = st.selectbox("Lanes_or_Medians", ["Two-way (divided with broken lines road marking)", "Undivided", "One way", "Other"])
        road_surface_type = st.selectbox("Road_surface_type", ["Asphalt roads", "Gravel roads", "Earth roads", "Other"])
        road_surface_conditions = st.selectbox("Road_surface_conditions", ["Dry", "Wet or damp", "Snow", "Flood over 3cm. deep"])

    type_of_collision = st.selectbox("Type_of_collision", [
        "Vehicle with vehicle collision",
        "Collision with roadside-parked vehicles",
        "Collision with roadside objects",
        "Overturning",
        "Collision with animals",
        "Fall from vehicles",
        "Other"
    ])

    cause_of_accident = st.selectbox("Cause_of_accident", [
        "Overspeed", "Overturning", "Moving Backward", "Driving carelessly", "Other"
    ])

    pedestrian_movement = st.selectbox("Pedestrian_movement", [
        "Not a Pedestrian", "Crossing", "Walking along roadside", "Other"
    ])

    casualty_class = st.selectbox("Casualty_class", ["na", "Driver", "Passenger", "Pedestrian"])
    casualty_severity = st.selectbox("Casualty_severity", ["na", "Slight", "Serious", "Fatal"])
    age_band_of_casualty = st.selectbox("Age_band_of_casualty", ["na", "0-5", "6-15", "16-30", "31-50", "51-70", "71+" ])
    work_of_casuality = st.selectbox("Work_of_casuality", ["Driver", "Passenger", "Pedestrian"])
    fitness_of_casuality = st.selectbox("Fitness_of_casuality", ["Normal", "Injured", "Under influence", "Other"])
    time_of_accident = st.text_input("Time of accident (HH or HH:MM)", "12:00")

    submit = st.form_submit_button("Predict")

# Prediction
if submit:
    input_data = pd.DataFrame({
        'Day_of_week': [day_of_week],
        'Age_band_of_driver': [age_band_of_driver],
        'Sex_of_driver': [sex_of_driver],
        'Educational_level': [educational_level],
        'Vehicle_driver_relation': [vehicle_driver_relation],
        'Driving_experience': [driving_experience],
        'Type_of_vehicle': ['Automobile'],  # default if not in form
        'Owner_of_vehicle': ['Owner'],      # default if not in form
        'Service_year_of_vehicle': [service_year_of_vehicle],
        'Defect_of_vehicle': [defect_of_vehicle],
        'Area_accident_occured': [area_accident_occured],
        'Lanes_or_Medians': [lanes_or_medians],
        'Road_alignment': [road_alignment],
        'Types_of_Junction': [types_of_junction],
        'Road_surface_type': [road_surface_type],
        'Road_surface_conditions': [road_surface_conditions],
        'Light_conditions': [light_conditions],
        'Weather_conditions': [weather_conditions],
        'Type_of_collision': [type_of_collision],
        'Number_of_casualties': [number_of_casualties],
        'Vehicle_movement': [vehicle_movement],
        'Casualty_class': [casualty_class],
        'Casualty_severity': [casualty_severity],
        'Age_band_of_casualty': [age_band_of_casualty],
        'Work_of_casuality': [work_of_casuality],
        'Fitness_of_casuality': [fitness_of_casuality],
        'Pedestrian_movement': [pedestrian_movement],
        'Cause_of_accident': [cause_of_accident],
        'Time_of_accident': [time_of_accident]
    })

    # Prediction
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Accident Severity: **{prediction}**")
