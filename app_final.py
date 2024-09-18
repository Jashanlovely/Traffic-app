import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from streamlit_option_menu import option_menu
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(page_title="Dashboard", page_icon="🌟", layout="wide", initial_sidebar_state="expanded")

# Apply custom CSS for dark mode and background image
st.markdown("""
    <style>
    .css-18e3th9 {
        background-color: #0e1117;
    }
    .css-1d391kg {
        color: #fafafa;
    }
    .css-1cpxqw2 a {
        color: #fafafa;
    }
    .css-1aumxhk a {
        color: #fafafa;
    }
    .css-qri22k a {
        color: #fafafa;
    }
    .css-1d391kg:before {
        content: "";
        background: url('dark-gradient.jpg') no-repeat center center fixed;
        background-size: cover;
        position: absolute;
        top: 0;
        left: 0;
        z-index: -1;
        height: 100%;
        width: 100%;
        opacity: 0.2;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Public Health Data", "Transportation Data", "Other","Hospital Admission Prediction",'Air Qulaity Index Prediction','Traffic Volume Prediction','CO2 Emission Prediction'],
                           icons=["activity", "truck", "info"],
                           menu_icon="cast", default_index=0)

# Main application logic
if selected == "Public Health Data":
    st.title("Public Health Data Dashboard")

    # Load the dataset
    data = pd.read_csv('HDHI Admission data.csv')

    # Convert 'Date of Admission' to datetime format
    data['Date of Admission'] = pd.to_datetime(data['Date of Admission'], errors='coerce')
    data['Type of Admission (Emergency/Outpatient)'].replace(
        {'E': 'Emergency', 'O': 'Outpatient'},
        inplace=True
    )

    # Randomly update 30% of 'Emergency' to 'Inpatient'
    emergency_indices = data[data['Type of Admission (Emergency/Outpatient)'] == 'Emergency'].sample(frac=0.3).index
    data.loc[emergency_indices, 'Type of Admission (Emergency/Outpatient)'] = 'Inpatient'

    # Display Hospital Statistics
    st.title('Hospital Statistics')

    # Add a month filter in the sidebar
    selected_month = st.sidebar.selectbox('Select a Month', ['Overall'] + list(range(1, 13)), 0)

    # Filter data based on the selected month or show overall statistics
    if selected_month == 'Overall':
        filtered_data = data
    else:
        filtered_data = data[data['Date of Admission'].dt.month == selected_month]

    # Display hospital statistics in the main part of the dashboard
    total_patients = len(filtered_data)
    outcome_distribution = filtered_data['Outcome'].value_counts()

    # Calculate the lengths of different outcome categories
    discharged_length = str(100 - int(round((len(filtered_data[filtered_data['Outcome'] == 'EXPIRY']) / total_patients) * 100, 0))) + "%"
    expiry_length = str(int(round((len(filtered_data[filtered_data['Outcome'] == 'EXPIRY']) / total_patients) * 100, 0))) + "%"

    col1, col2, col3= st.columns(3)

    # Define a function to create a decorated box
    def create_box(title, value, col):
        box_style = "border: 2px solid #4682B4; border-radius: 10px; padding: 10px; background-color: #E0EBF5; width: 150px; height:150px;"
        col.markdown(
            f"""
            <div style="{box_style}">
                <h3 style="color: #4682B4;">{title}</h3>
                <p style="font-size: 20px; font-weight: bold; color: #1E90FF;">{value}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Create decorated boxes for total patients, discharged patients, and mortality rate
    create_box("Total Patients", total_patients, col1)
    create_box("Discharge Rate", discharged_length, col2)
    create_box("Mortality Rate", expiry_length, col3)

    # Comorbidities Statistics
    comorbidity_columns = [
        'Diabetes Mellitus (DM)', 'Hypertension (HTN)', 'Prior Coronary Artery Disease (CAD)',
        'Prior Cardiomyopathy (CMP)', 'Chronic Kidney Disease (CKD)', 'Ejection Fraction (EF)',
        'Severe Anemia', 'Anemia', 'Stable Angina', 'Acute Coronary Syndrome (ACS)',
        'ST-Elevation Myocardial Infarction (STEMI)', 'Atypical Chest Pain', 'Heart Failure',
        'Heart Failure with Reduced Ejection Fraction (HFREF)', 'Heart Failure with Normal Ejection Fraction (HFNEF)',
        'Valvular Heart Disease', 'Complete Heart Block (CHB)', 'Sick Sinus Syndrome (SSS)',
        'Acute Kidney Injury (AKI)', 'Cerebrovascular Accident (CVA) - Infarct', 'Cerebrovascular Accident (CVA) - Bleed',
        'Atrial Fibrillation (AF)', 'Ventricular Tachycardia (VT)', 'Paroxysmal Supraventricular Tachycardia (PSVT)',
        'Congenital Heart Disease', 'Urinary Tract Infection (UTI)', 'Neurocardiogenic Syncope',
        'Orthostatic Hypotension', 'Infective Endocarditis', 'Deep Vein Thrombosis (DVT)',
        'Cardiogenic Shock', 'Shock', 'Pulmonary Embolism'
    ]

    # Sum the occurrences of each comorbidity
    comorbidity_counts = filtered_data[comorbidity_columns].sum(numeric_only=True)

    top_07_comorbidities = pd.DataFrame(comorbidity_counts.sort_values(ascending=False).head(7)).reset_index()
    top_07_comorbidities.columns=['Comorbidities','Count']
    comorbidity_columns_of_interest = top_07_comorbidities['Comorbidities'].to_list()

    selected_comorbidities_data = filtered_data[comorbidity_columns_of_interest]

    # Create a pivot table to get counts
    comorbidity_counts = selected_comorbidities_data.sum()

    st.title('Comorbidities Statistics')

    # Create a tree map using plotly express
    fig = px.treemap(names=comorbidity_counts.index, parents=[''] * len(comorbidity_counts),
                    values=comorbidity_counts.values,
                    color=comorbidity_counts.values, color_continuous_scale='darkmint',  # Use 'darkmint' for darker colors
                    labels={'value': 'Count'},
                    custom_data=[comorbidity_counts.index, comorbidity_counts.values])

    # Update the layout to display values and names in bold
    fig.update_traces(textinfo='label+value', hovertemplate='<b>%{label}</b><br>Count: %{value}')

    st.plotly_chart(fig)

    # Rename columns to match the dataset description
    data.rename(columns={
        'SNO': 'Serial Number',
        'MRD No.': 'MRD Number',
        'D.O.A': 'Date of Admission',
        'D.O.D': 'Date of Discharge',
        'AGE': 'Age',
        'GENDER': 'Gender',
        'RURAL': 'Locality (Rural/Urban)',
        'TYPE OF ADMISSION-EMERGENCY/OPD': 'Type of Admission (Emergency/Outpatient)',
        'month year': 'Month Year',
        'DURATION OF STAY': 'Duration of Stay',
        'duration of intensive unit stay': 'Duration of Intensive Unit Stay',
        'OUTCOME': 'Outcome',
        'SMOKING': 'Smoking',
        'ALCOHOL': 'Alcohol',
        'DM': 'Diabetes Mellitus (DM)',
        'HTN': 'Hypertension (HTN)',
        'CAD': 'Prior Coronary Artery Disease (CAD)',
        'PRIOR CMP': 'Prior Cardiomyopathy (CMP)',
        'CKD': 'Chronic Kidney Disease (CKD)',
        'HB': 'Hemoglobin (HB)',
        'TLC': 'Total Lymphocyte Count (TLC)',
        'PLATELETS': 'Platelets',
        'GLUCOSE': 'Glucose',
        'UREA': 'Urea',
        'CREATININE': 'Creatinine',
        'BNP': 'Brain Natriuretic Peptide (BNP)',
        'RAISED CARDIAC ENZYMES': 'Raised Cardiac Enzymes',
        'EF': 'Ejection Fraction (EF)',
        'SEVERE ANAEMIA': 'Severe Anemia',
        'ANAEMIA': 'Anemia',
        'STABLE ANGINA': 'Stable Angina',
        'ACS': 'Acute Coronary Syndrome (ACS)',
        'STEMI': 'ST-Elevation Myocardial Infarction (STEMI)',
        'ATYPICAL CHEST PAIN': 'Atypical Chest Pain',
        'HEART FAILURE': 'Heart Failure',
        'HFREF': 'Heart Failure with Reduced Ejection Fraction (HFREF)',
        'HFNEF': 'Heart Failure with Normal Ejection Fraction (HFNEF)',
        'VALVULAR': 'Valvular Heart Disease',
        'CHB': 'Complete Heart Block (CHB)',
        'SSS': 'Sick Sinus Syndrome (SSS)',
        'AKI': 'Acute Kidney Injury (AKI)',
        'CVA INFRACT': 'Cerebrovascular Accident (CVA) - Infarct',
        'CVA BLEED': 'Cerebrovascular Accident (CVA) - Bleed',
        'AF': 'Atrial Fibrillation (AF)',
        'VT': 'Ventricular Tachycardia (VT)',
        'PSVT': 'Paroxysmal Supraventricular Tachycardia (PSVT)',
        'CONGENITAL': 'Congenital Heart Disease',
        'UTI': 'Urinary Tract Infection (UTI)',
        'NEURO CARDIOGENIC SYNCOPE': 'Neurocardiogenic Syncope',
        'ORTHOSTATIC': 'Orthostatic Hypotension',
        'INFECTIVE ENDOCARDITIS': 'Infective Endocarditis',
        'DVT': 'Deep Vein Thrombosis (DVT)',
        'CARDIOGENIC SHOCK': 'Cardiogenic Shock',
        'SHOCK': 'Shock',
        'PULMONARY EMBOLISM': 'Pulmonary Embolism',
        'CHEST INFECTION': 'Chest Infection'
    }, inplace=True)

    st.sidebar.title('Dashboard Options')

    # Sidebar filter options
    selected_gender = st.sidebar.selectbox('Select Gender', data['Gender'].unique())
    selected_age = st.sidebar.slider('Select Age', float(data['Age'].min()), float(data['Age'].max()), 
                                      (float(data['Age'].min()), float(data['Age'].max())))

    # Filter the data based on user selections
    filtered_data = data[(data['Gender'] == selected_gender) & 
                         (data['Age'] >= selected_age[0]) & 
                         (data['Age'] <= selected_age[1])]

    # Display the filtered data
    st.write('### Filtered Data')
    st.write(filtered_data)

    # Create a 2x2 grid layout for plots using Plotly
    # Plot 1: Bar chart for Outcome Counts
    st.subheader('Outcome Counts')
    bar_data = filtered_data['Outcome'].value_counts().reset_index()
    bar_data.columns = ['Outcome', 'Count']
    fig_bar = px.bar(bar_data, x='Outcome', y='Count', title='Outcome Counts', 
                     labels={'Outcome': 'Outcome', 'Count': 'Count'}, color='Count')
    st.plotly_chart(fig_bar)

    # Plot 2: Histogram for Age Distribution
    st.subheader('Age Distribution')
    fig_hist = px.histogram(filtered_data, x='Age', nbins=20, title='Age Distribution', 
                            labels={'Age': 'Age'}, marginal='box')
    st.plotly_chart(fig_hist)

    # Plot 3: Line chart for Admission Count Over Time
    st.subheader('Admission Count Over Time')
    line_data = filtered_data.groupby('Date of Admission').size().reset_index(name='Count')
    line_data['Date of Admission'] = pd.to_datetime(line_data['Date of Admission'])
    fig_line = px.line(line_data, x='Date of Admission', y='Count', title='Admission Count Over Time', 
                       labels={'Date of Admission': 'Date of Admission', 'Count': 'Count'})
    st.plotly_chart(fig_line)

    # Plot 4: Box plot for Age by Outcome
    st.subheader('Age Distribution by Outcome')
    fig_box = px.box(filtered_data, x='Outcome', y='Age', title='Age Distribution by Outcome', 
                     labels={'Outcome': 'Outcome', 'Age': 'Age'})
    st.plotly_chart(fig_box)

        # Air Quality Data Analysis
    st.title('Air Quality Data Analysis')

        # Load the air quality dataset
    air_quality_data = pd.read_csv('city_day.csv')


    # Plot the distribution of AQI bucket categories
    st.subheader('Distribution of AQI Bucket Categories')
    fig = px.bar(air_quality_data, x='AQI_Bucket', color='AQI_Bucket', 
                 title='Distribution of AQI Bucket Categories',
                 labels={'AQI_Bucket': 'AQI Bucket', 'count': 'Number of Samples'})
    st.plotly_chart(fig)

    # Plot the distribution of AQI values
    st.subheader('Distribution of AQI Values')
    fig = px.histogram(air_quality_data, x='AQI', nbins=30, 
                       title='Distribution of AQI Values',
                       labels={'AQI': 'AQI Value'})
    st.plotly_chart(fig)

    # Air Quality by City
    st.subheader('Average AQI by City')
    fig = px.bar(air_quality_data.groupby('City')['AQI'].mean().reset_index(), 
                 x='City', y='AQI',
                 title='Average AQI by City',
                 labels={'City': 'City', 'AQI': 'Average AQI'})
    st.plotly_chart(fig)

    # Box Plot for City
    st.subheader('AQI Distribution by City')
    fig = px.box(air_quality_data, x='City', y='AQI',
                 title='AQI Distribution by City',
                 labels={'City': 'City', 'AQI': 'AQI'})
    st.plotly_chart(fig)


elif selected == "Transportation Data":
    st.title("Transportation Data")

    # Air Quality Data Analysis
    st.title('Traffic Volume Analysis')

    # Load dataset
    data1 = pd.read_csv('Train_traffic.csv')

    # Define simplified mapping for weather descriptions
    weather_simplification = {
        'thunderstorm': 'thunderstorm',
        'thunderstorm with drizzle': 'thunderstorm',
        'thunderstorm with heavy rain': 'thunderstorm',
        'thunderstorm with light drizzle': 'thunderstorm',
        'thunderstorm with light rain': 'thunderstorm',
        'thunderstorm with rain': 'thunderstorm',
        'proximity thunderstorm': 'thunderstorm',
        'proximity thunderstorm with drizzle': 'thunderstorm',
        'proximity thunderstorm with rain': 'thunderstorm',
        'squalls': 'thunderstorm',
        
        'drizzle': 'drizzle',
        'light intensity drizzle': 'drizzle',
        'shower drizzle': 'drizzle',
        'freezing rain': 'drizzle',
        'heavy intensity drizzle': 'drizzle',
        
        'fog': 'fog',
        'mist': 'fog',
        'haze': 'fog',
        
        'light rain': 'rain',
        'moderate rain': 'rain',
        'heavy intensity rain': 'rain',
        'proximity shower rain': 'rain',
        'very heavy rain': 'rain',
        'light rain and snow': 'rain/snow',
        'light intensity shower rain': 'rain/snow',
        
        'snow': 'snow',
        'heavy snow': 'snow',
        'shower snow': 'snow',
        'light shower snow': 'snow',
        
        'overcast clouds': 'cloudy',
        'scattered clouds': 'cloudy',
        'few clouds': 'cloudy',
        'broken clouds': 'cloudy',
        
        'sky is clear': 'clear',
        
        'smoke': 'smoke',
        'sleet': 'sleet'
    }


    # Apply the mapping to simplify weather descriptions
    data1['weather_description_simplified'] = data1['weather_description'].map(weather_simplification)


    # Bar Chart Showing Weather Conditions
    st.subheader('Weather Conditions Frequency')
    weather_counts = data1['weather_type'].value_counts().reset_index()
    weather_counts.columns = ['Weather Conditions', 'Frequency']
    fig_weather = px.bar(weather_counts, x='Weather Conditions', y='Frequency', 
                         title='Weather Conditions Frequency', 
                         labels={'Weather Conditions': 'Weather Conditions', 'Frequency': 'Frequency'},
                         color='Frequency')
    st.plotly_chart(fig_weather)

    # Distribution of Traffic Volume
    st.subheader('Distribution of Traffic Volume')
    fig_traffic_volume = px.histogram(data1, x='traffic_volume', nbins=50, 
                                       title='Distribution of Traffic Volume', 
                                       labels={'traffic_volume': 'Traffic Volume'},
                                       marginal='box')
    st.plotly_chart(fig_traffic_volume)

    # Traffic volume by hour
    data1['hour'] = pd.to_datetime(data1['date_time']).dt.hour
    st.subheader('Traffic Volume by Hour')
    fig_hour = px.box(data1, x='hour', y='traffic_volume', 
                      title='Traffic Volume by Hour', 
                      labels={'hour': 'Hour of Day', 'traffic_volume': 'Traffic Volume'})
    st.plotly_chart(fig_hour)

    # Traffic volume by day of the week
    data1['day_of_week'] = pd.to_datetime(data1['date_time']).dt.dayofweek
    st.subheader('Traffic Volume by Day of the Week')
    fig_day_of_week = px.box(data1, x='day_of_week', y='traffic_volume', 
                              title='Traffic Volume by Day of the Week', 
                              labels={'day_of_week': 'Day of the Week', 'traffic_volume': 'Traffic Volume'})
    st.plotly_chart(fig_day_of_week)

    # Traffic volume by weather type
    st.subheader('Traffic Volume by Weather Type')
    fig_weather_type = px.box(data1, x='weather_type', y='traffic_volume', 
                               title='Traffic Volume by Weather Type', 
                               labels={'weather_type': 'Weather Type', 'traffic_volume': 'Traffic Volume'})
    st.plotly_chart(fig_weather_type)

    # Traffic volume vs temperature
    st.subheader('Traffic Volume vs Temperature')
    fig_temp = px.scatter(data1, x='temperature', y='traffic_volume', 
                          title='Traffic Volume vs Temperature', 
                          labels={'temperature': 'Temperature', 'traffic_volume': 'Traffic Volume'})
    st.plotly_chart(fig_temp)

    
    # Air Quality Data Analysis
    st.title('Vehicle CO2 Emission Analysis')

    # Load the air quality dataset
    air_quality_data = pd.read_csv('city_day.csv')
    file_path = 'CO2 Emissions_Canada.csv'  # Replace with the path to your dataset
    data = pd.read_csv(file_path)

    # Rename Columns
    data.rename(columns={ 
        'Make': 'make',
        'Model': 'model',
        'Vehicle Class': 'vehicle_class',
        'Engine Size(L)': 'engine_size',
        'Cylinders': 'cylinders',
        'Transmission': 'transmission',
        'Fuel Type': 'fuel_type',
        'Fuel Consumption City (L/100 km)': 'fuel_cons_city',
        'Fuel Consumption Hwy (L/100 km)': 'fuel_cons_hwy',
        'Fuel Consumption Comb (L/100 km)': 'fuel_cons_comb',
        'Fuel Consumption Comb (mpg)': 'fuel_cons_comb_mpg',
        'CO2 Emissions(g/km)': 'co2'
    }, inplace=True)

    # Vertical Bar Graphs for Categorical Features
    cat_features = ['make', 'vehicle_class', 'transmission', 'fuel_type']

    for column in cat_features:
        fig = px.bar(data_frame=data, x=column, title=f'Bar Graph of {column}', 
                     labels={column: column, 'count': 'Count'}, 
                     category_orders={column: data[column].value_counts().index})
        st.plotly_chart(fig)

    # Top 20 Models Bar Chart
    top_models = data['model'].value_counts().head(20)
    fig = px.bar(x=top_models.index, y=top_models.values, title='Top 20 Models', 
                 labels={'x': 'Model', 'y': 'Number of Cars'})
    st.plotly_chart(fig)

    # Mean CO2 Emission by Categorical Features
    for column in cat_features:
        grouped_data = data.groupby(column)['co2'].mean().reset_index()
        grouped_data_sorted = grouped_data.sort_values(by='co2', ascending=False)
        fig = px.bar(data_frame=grouped_data_sorted, x=column, y='co2', 
                     title=f'Mean CO2 Emission by {column}', 
                     labels={column: column, 'co2': 'Mean CO2 Emission'})
        st.plotly_chart(fig)

    # Histograms for Numerical Variables (as Bar Charts)
    numerical_df = data.select_dtypes(include=['number'])

    for var in numerical_df.columns:
        fig = px.histogram(data_frame=data, x=var, title=f'Distribution of {var}', 
                           nbins=20, labels={var: var})
        st.plotly_chart(fig)

    # Bar Chart of Fuel Consumption (City and Highway)
    fuel_consumption = data[['fuel_cons_hwy', 'fuel_cons_city']].melt(var_name='Type', value_name='Consumption')
    fig = px.bar(fuel_consumption, x='Type', y='Consumption', 
                 title='Fuel Consumption in City and Highway', 
                 labels={'Type': 'Fuel Consumption Type', 'Consumption': 'Consumption (L/100 km)'}, 
                 barmode='group')
    st.plotly_chart(fig)

    # Average CO2 emissions by engine size (binned)
    data['engine_size_bin'] = pd.cut(data['engine_size'], bins=[0, 1, 2, 3, 4, 5, 6, 7], 
                                      labels=['0-1L', '1-2L', '2-3L', '3-4L', '4-5L', '5-6L', '6+L'])
    engine_size_avg_co2 = data.groupby('engine_size_bin')['co2'].mean().reset_index()
    fig_engine = px.bar(data_frame=engine_size_avg_co2, x='engine_size_bin', y='co2', 
                        title='Average CO2 Emissions by Engine Size', 
                        labels={'engine_size_bin': 'Engine Size (L)', 'co2': 'Average CO2 Emissions (g/km)'})
    st.plotly_chart(fig_engine)

    # Average CO2 emissions by number of cylinders
    cylinder_avg_co2 = data.groupby('cylinders')['co2'].mean().reset_index()
    fig_cylinders = px.bar(data_frame=cylinder_avg_co2, x='cylinders', y='co2', 
                           title='Average CO2 Emissions by Number of Cylinders', 
                           labels={'cylinders': 'Number of Cylinders', 'co2': 'Average CO2 Emissions (g/km)'})
    st.plotly_chart(fig_cylinders)


elif selected == "Hospital Admission Prediction":
    st.title("Hospital Admission Count Prediction")

    # Assuming you have already trained and saved your model as 'hospital_admission_model.pkl'
    model_path = 'xgb_model_admission.pkl'  # Replace with your model path

    # Load the trained model
    with open(model_path, 'rb') as file:
        hospital_admission_model = pickle.load(file)


    # Direct user inputs for each feature
    month = st.number_input("Month (YYYYMM)", value=0.0)
    month_year = st.number_input("Month Year", value=0.0)
    duration_of_stay = st.number_input("Duration of Stay (days)", value=0.0)
    hemoglobin = st.number_input("Hemoglobin (HB)", value=0.0)
    tlc = st.number_input("Total Lymphocyte Count (TLC)", value=0.0)
    platelets = st.number_input("Platelets", value=0.0)
    type_of_admission = st.number_input("Type of Admission", value=0)
    age = st.number_input("Age", value=0)

    # Convert type of admission to numerical (0 for Emergency, 1 for Outpatient)
    #type_of_admission_numeric = 0 if type_of_admission == "Emergency" else 1

    # Prediction button
    if st.button("Predict Admission Count"):
        # Prepare the input data for prediction
        input_data = pd.DataFrame({
            'Month': [month],
            'Month Year': [month_year],
            'Duration of Stay': [duration_of_stay],
            'Hemoglobin (HB)': [hemoglobin],
            'Total Lymphocyte Count (TLC)': [tlc],
            'Platelets': [platelets],
            'Type of Admission (Emergency/Outpatient)': [type_of_admission],
            'Age': [age]
        })

        # Make the prediction
        prediction = hospital_admission_model.predict(input_data)

        # Display the result
        st.write(f'Predicted Admission Count: {int(prediction[0])}')  # Convert prediction to int for discrete count


elif selected == "Air Qulaity Index Prediction":
    st.title("Air Qulaity Index Prediction")
    # Load the trained model from the pickle file
    model_path = 'xgb_model_aqi.pkl'  # Replace with your model path
    with open(model_path, 'rb') as file:
        air_quality_model = pickle.load(file)

    # Direct user inputs for each feature
    pm2_5 = st.number_input("PM2.5", value=0.0)
    pm10 = st.number_input("PM10", value=0.0)
    no = st.number_input("NO", value=0.0)
    no2 = st.number_input("NO2", value=0.0)
    nox = st.number_input("NOx", value=0.0)
    nh3 = st.number_input("NH3", value=0.0)
    co = st.number_input("CO", value=0.0)
    so2 = st.number_input("SO2", value=0.0)
    o3 = st.number_input("O3", value=0.0)
    benzene = st.number_input("Benzene", value=0.0)
    toluene = st.number_input("Toluene", value=0.0)
    xylene = st.number_input("Xylene", value=0.0)
   

    # Prepare input data
    input_data = pd.DataFrame({
        'PM2.5': [pm2_5],
        'PM10': [pm10],
        'NO': [no],
        'NO2': [no2],
        'NOx': [nox],
        'NH3': [nh3],
        'CO': [co],
        'SO2': [so2],
        'O3': [o3],
        'Benzene': [benzene],
        'Toluene': [toluene],
        'Xylene': [xylene]
    })

    # Predict AQI using the model
    if st.button("Predict AQI"):
        prediction = air_quality_model.predict(input_data)
        predicted_aqi = int(prediction[0])  # Ensure it's treated as an integer

        # Map numerical prediction to categorical labels
        category_mapping = {
            0: "Good",
            1: "Moderate",
            2: "Poor",
            3: "Satisfactory",
            4: "Severe",
            5: "Very Poor"
        }
        
        # Get the corresponding AQI category
        aqi_category = category_mapping.get(predicted_aqi, "Unknown")

        st.write(f"Predicted AQI Encoded Value: {predicted_aqi}")
        st.write(f"AQI Category: {aqi_category}")

elif selected == "Traffic Volume Prediction":
    st.title("Traffic Volume Prediction")

    # Load the trained model from the pickle file
    model_path = 'traffic_rf_model.pkl'  # Replace with your actual model path
    with open(model_path, 'rb') as file:
        traffic_model = pickle.load(file)

    # Direct user inputs for each feature
    is_holiday = st.text_input("Is Holiday", value='0')  # User inputs as string or number
    air_pollution_index = st.text_input("Air Pollution Index", value='0')  # User inputs as string or number
    humidity = st.text_input("Humidity", value='0')  # User inputs as string or number
    wind_speed = st.text_input("Wind Speed", value='0')  # User inputs as string or number
    wind_direction = st.text_input("Wind Direction", value='0')  # User inputs as string or number
    visibility_in_miles = st.text_input("Visibility in Miles", value='0')  # User inputs as string or number
    dew_point = st.text_input("Dew Point", value='0')  # User inputs as string or number
    temperature = st.number_input("Temperature", min_value=-100.0, step=0.1)  # User inputs temperature
    rain_p_h = st.text_input("Rain (p/h)", value='0')  # User inputs as string or number
    snow_p_h = st.text_input("Snow (p/h)", value='0')  # User inputs as string or number
    clouds_all = st.text_input("Clouds (%)", value='0')  # User inputs as string or number
    weather_type = st.text_input("Weather Type", value='0')  # User inputs as string or number

    # Prepare feature list for prediction
    features = [
        int(is_holiday) if is_holiday else 0,
        int(air_pollution_index) if air_pollution_index else 0,
        int(humidity) if humidity else 0,
        int(wind_speed) if wind_speed else 0,
        int(wind_direction) if wind_direction else 0,
        int(visibility_in_miles) if visibility_in_miles else 0,
        int(dew_point) if dew_point else 0,
        float(temperature),
        int(rain_p_h) if rain_p_h else 0,
        int(snow_p_h) if snow_p_h else 0,
        int(clouds_all) if clouds_all else 0,
        int(weather_type) if weather_type else 0
    ]

    # Convert features into a DataFrame for prediction
    input_data = pd.DataFrame([features], columns=[
        'is_holiday', 'air_pollution_index', 'humidity', 'wind_speed', 
        'wind_direction', 'visibility_in_miles', 'dew_point', 
        'temperature', 'rain_p_h', 'snow_p_h', 'clouds_all', 'weather_type'
    ])



    if st.button("Predict Traffic Volume"):
        prediction = traffic_model.predict(input_data)
        predicted_category = prediction[0]

        # Map numerical prediction to categorical labels
        category_mapping = {0: "High", 1: "Low", 2: "Medium"}
        category_label = category_mapping.get(predicted_category, "Unknown")

        st.write(f"Predicted Traffic Volume Category: {category_label} (Encoded Value: {predicted_category})")

elif selected == "CO2 Emission Prediction":
    st.title("CO2 Emission Prediction")
    # Load the trained model from the pickle file
    model_path = 'co2_xgb_model.pkl'  # Replace with your model path
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Direct user inputs for each feature
    make = st.text_input('Make', value='')  # User inputs make as a string or number
    model_input = st.text_input('Model', value='')  # User inputs model as a string or number
    vehicle_class = st.text_input('Vehicle Class', value='')  # User inputs vehicle class as a string or number
    engine_size = st.number_input('Engine Size (L)', min_value=0.0, step=0.1)  # User inputs engine size
    cylinders = st.number_input('Number of Cylinders', min_value=1, step=1)  # User inputs number of cylinders
    transmission = st.text_input('Transmission Type', value='')  # User inputs transmission as a string or number
    fuel_type = st.text_input('Fuel Type', value='')  # User inputs fuel type as a string or number
    fuel_cons_city = st.number_input('Fuel Consumption City (L/100 km)', min_value=0.0, step=0.1)  # User inputs city fuel consumption
    fuel_cons_hwy = st.number_input('Fuel Consumption Highway (L/100 km)', min_value=0.0, step=0.1)  # User inputs highway fuel consumption
    fuel_cons_comb = st.number_input('Fuel Consumption Combined (L/100 km)', min_value=0.0, step=0.1)  # User inputs combined fuel consumption
    fuel_cons_comb_mpg = st.number_input('Fuel Consumption Combined (mpg)', min_value=0, step=1)  # User inputs combined fuel consumption in mpg

    # Convert input values to appropriate types and handle empty inputs
    input_data = pd.DataFrame({
        'make': [int(make) if make else 0],
        'model': [int(model_input) if model_input else 0],
        'vehicle_class': [int(vehicle_class) if vehicle_class else 0],
        'engine_size': [engine_size],
        'cylinders': [cylinders],
        'transmission': [int(transmission) if transmission else 0],
        'fuel_type': [int(fuel_type) if fuel_type else 0],
        'fuel_cons_city': [fuel_cons_city],
        'fuel_cons_hwy': [fuel_cons_hwy],
        'fuel_cons_comb': [fuel_cons_comb],
        'fuel_cons_comb_mpg': [fuel_cons_comb_mpg]
    })

    # Predict CO2 emissions using the model
    if st.button('Predict CO2 Emissions'):
        prediction = model.predict(input_data)
        st.success(f'Predicted CO2 Emissions: {prediction[0]:.2f} g/km')
    
# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<footer style='text-align: center;'>© 2024 Your Company Name</footer>", unsafe_allow_html=True)
