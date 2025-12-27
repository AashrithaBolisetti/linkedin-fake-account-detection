import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(page_title="LinkedIn Fake Account Detector", layout="centered")

@st.cache_resource
def train_model():
    # Load dataset
    dataset_directory = "LinkedIn_Dataset.pcl"
    dataset = pd.read_pickle(dataset_directory)

    # Preprocessing (Simplified for deployment stability)
    # Convert dict type columns to string
    for col in dataset.columns:
        if dataset[col].apply(lambda x: isinstance(x, dict)).any():
            dataset[col] = dataset[col].apply(str)

    # Fill missing values
    dataset['Full Name'] = dataset['Full Name'].fillna('Unknown')
    dataset['Workplace'] = dataset['Workplace'].fillna('Unknown')
    dataset['Location'] = dataset['Location'].fillna('Unknown')

    # For the app, we focus on the core numeric/boolean features to avoid 
    # the 32,000+ column explosion caused by dummy encoding every name.
    # We will pick the most important features identified in your notebook.
    features = ['Number of Experiences', 'Number of Educations', 'Number of Skills', 
                'Number of Interests', 'Number of Activities']
    
    # Convert these columns to numeric, forcing errors to 0
    for f in features:
        dataset[f] = pd.to_numeric(dataset[f], errors='coerce').fillna(0)

    X = dataset[features]
    y = dataset['Label']

    # Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    return clf, scaler, features

# Load model and scaler
with st.spinner("Initializing Model..."):
    model, scaler, feature_cols = train_model()

# UI Layout
st.title("🛡️ LinkedIn Fake Account Detector")
st.write("Enter profile details below to check for authenticity.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        exp = st.number_input("Number of Experiences", min_value=0, value=2)
        edu = st.number_input("Number of Educations", min_value=0, value=1)
        skills = st.number_input("Number of Skills", min_value=0, value=5)
    
    with col2:
        interests = st.number_input("Number of Interests", min_value=0, value=4)
        activities = st.number_input("Number of Activities", min_value=0, value=1)
    
    submit = st.form_submit_button("Detect Account Type")

if submit:
    # Prepare input data
    input_data = pd.DataFrame([[exp, edu, skills, interests, activities]], 
                              columns=feature_cols)
    
    # Scale and Predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    # Display Results
    st.divider()
    if prediction == 1:
        st.error("### Result: Potential Fake Account Detected")
        st.warning("This profile matches patterns often found in automated or fake accounts.")
    else:
        st.success("### Result: Likely Authentic Account")
        st.info("This profile matches patterns of a standard LinkedIn user.")

st.sidebar.info("This app uses a RandomForestClassifier trained on LinkedIn profile metadata.")
