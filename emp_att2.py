import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import time

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load saved objects
@st.cache_resource
def load_model():
    try:
        saved_objects = joblib.load("employee_attrition_model.joblib")
        return saved_objects
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'employee_attrition_model.joblib' is in the correct directory.")
        return None

saved_objects = load_model()

if saved_objects:
    model = saved_objects["model"]
    encoder = saved_objects["encoder"]
    scaler = saved_objects["scaler"]
    numerical_cols = saved_objects["numerical_cols"]
    categorical_cols = saved_objects["categorical_cols"]
    encoded_cols = saved_objects["encoded_cols"]

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: scale(1.05);
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .high-risk {
        background-color: #ffebe6;
        border-left: 5px solid #e74c3c;
    }
    .low-risk {
        background-color: #e6f4ea;
        border-left: 5px solid #2ecc71;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .uploaded-file {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">👥 Employee Attrition Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; color: #7f8c8d; font-size: 1.2rem;'>
Predict employee attrition risk using Logistic Regression. Upload your data to get started.
</p>
""", unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1001/1001040.png", width=100)
    st.title("Navigation")
    app_mode = st.radio("Choose a mode:", ["📁 Upload Data", "ℹ️ About"])
    
    st.markdown("---")
    st.markdown("### Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.7, 0.05,
                                   help="Set the minimum probability threshold for high-risk classification")
    
    st.markdown("---")
    st.info("""
    This app predicts employee attrition risk using a Logistic Regression model. 
    Upload a CSV file with employee data to get predictions.
    """)

# Main content
if app_mode == "📁 Upload Data":
    st.markdown('<h2 class="sub-header">Upload Employee Data</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["csv"], 
                                    help="Upload a CSV file containing employee data")
    
    if uploaded_file is not None:
        with st.spinner("Processing your data..."):
            try:
                df = pd.read_csv(uploaded_file)
                
                # Display file info
                st.markdown(f'<div class="uploaded-file">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Employees", len(df))
                with col2:
                    st.metric("Features", len(df.columns))
                with col3:
                    st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Check if all required columns are present
                required_cols = numerical_cols + categorical_cols
                missing_cols = set(required_cols) - set(df.columns)
                
                if missing_cols:
                    st.error(f"The uploaded file is missing the following required columns: {', '.join(missing_cols)}")
                else:
                    # Data preview with expander
                    with st.expander("Preview Uploaded Data", expanded=True):
                        st.dataframe(df.head(10))
                    
                    # Preprocessing and prediction
                    if st.button("Run Attrition Analysis", type="primary"):
                        with st.spinner("Analyzing data and generating predictions..."):
                            # Simulate processing time for better UX
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i in range(100):
                                progress_bar.progress(i + 1)
                                status_text.text(f"Processing... {i+1}%")
                                time.sleep(0.01)
                            
                            # EXACTLY REPLICATE YOUR TRAINING PREPROCESSING
                            # Step 1: OneHotEncoding for categorical variables
                            df_encoded = encoder.transform(df[categorical_cols])
                            
                            # Convert to DataFrame with proper column names
                            df_encoded_df = pd.DataFrame(df_encoded, columns=encoded_cols, index=df.index)
                            
                            # Step 2: MinMax Scaling for numerical variables
                            df_scaled = scaler.transform(df[numerical_cols])
                            df_scaled_df = pd.DataFrame(df_scaled, columns=numerical_cols, index=df.index)
                            
                            # Step 3: Combine encoded categorical and scaled numerical features
                            df_processed = pd.concat([df_scaled_df, df_encoded_df], axis=1)
                            
                            # Ensure the column order matches training
                            expected_columns = numerical_cols + encoded_cols
                            df_processed = df_processed[expected_columns]
                            
                            progress_bar.progress(70)
                            
                            # Prediction using your Logistic Regression model
                            predictions = model.predict(df_processed)
                            probabilities = model.predict_proba(df_processed)[:, 1]  # Probability of "Yes"
                            
                            # Add predictions to original dataframe
                            df["Attrition_Prediction"] = predictions
                            df["Attrition_Probability"] = probabilities
                            df["Risk_Level"] = np.where(df["Attrition_Probability"] > confidence_threshold, 
                                                      "High Risk", "Low Risk")
                            
                            status_text.text("Analysis complete!")
                            time.sleep(0.5)
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Display results
                            st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
                            
                            # Summary metrics
                            high_risk_count = (df["Risk_Level"] == "High Risk").sum()
                            low_risk_count = (df["Risk_Level"] == "Low Risk").sum()
                            attrition_rate = high_risk_count / len(df) * 100
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.metric("High Risk Employees", high_risk_count)
                                st.markdown('</div>', unsafe_allow_html=True)
                            with col2:
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.metric("Low Risk Employees", low_risk_count)
                                st.markdown('</div>', unsafe_allow_html=True)
                            with col3:
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.metric("Predicted Attrition Rate", f"{attrition_rate:.1f}%")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Visualization
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Risk distribution pie chart
                                risk_counts = df["Risk_Level"].value_counts()
                                fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                                            title="Risk Level Distribution",
                                            color=risk_counts.index,
                                            color_discrete_map={"High Risk": "#e74c3c", "Low Risk": "#2ecc71"})
                                fig.update_traces(textposition='inside', textinfo='percent+label')
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Probability distribution histogram
                                fig = px.histogram(df, x="Attrition_Probability", 
                                                  title="Attrition Probability Distribution",
                                                  nbins=20, color_discrete_sequence=['#3498db'])
                                fig.add_vline(x=confidence_threshold, line_dash="dash", line_color="red",
                                             annotation_text="Threshold", annotation_position="top")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show top 10 high-risk employees
                            st.markdown("### Top 10 High-Risk Employees")
                            high_risk_df = df[df["Risk_Level"] == "High Risk"].sort_values("Attrition_Probability", ascending=False).head(10)
                            st.dataframe(high_risk_df[['Gender','Age', 'Department', 'JobRole', 'MonthlyIncome', 'Attrition_Probability' ]])
                            
                            # Interactive results table
                            with st.expander("View All Predictions", expanded=False):
                                st.dataframe(df.sort_values("Attrition_Probability", ascending=False))
                            
                            # Download button
                            csv = df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="Download Predictions as CSV",
                                data=csv,
                                file_name="attrition_predictions.csv",
                                mime="text/csv",
                                key="download-csv"
                            )
                
            except Exception as e:
                st.error(f"Error processing the file: {str(e)}")
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload a CSV file to get started. The file should contain employee data with the required columns.")
        
        # Show required columns in an expandable section
        with st.expander("View Required Columns"):
            st.write("**Numerical Columns:**", numerical_cols)
            st.write("**Categorical Columns:**", categorical_cols)

elif app_mode == "ℹ️ About":
    st.markdown('<h2 class="sub-header">About This App</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem;">
    <h3>Employee Attrition Prediction System</h3>
    <p>This application uses a Logistic Regression model to predict employee attrition risk based on various factors.</p>
    
    <h3>Model Details</h3>
    <ul>
        <li><strong>Algorithm:</strong> Logistic Regression with class weighting</li>
        <li><strong>Preprocessing:</strong> OneHotEncoding for categorical variables, MinMax scaling for numerical variables</li>
        <li><strong>Key Features:</strong> OverTime, MonthlyIncome, Age, JobRole, BusinessTravel, and more</li>
    </ul>
    
    <h3>How to Use</h3>
    <ol>
        <li>Navigate to the <strong>Upload Data</strong> section</li>
        <li>Upload a CSV file containing employee data with the required columns</li>
        <li>Click the <strong>Run Attrition Analysis</strong> button</li>
        <li>Review the predictions and download results if needed</li>
    </ol>
    
    <h3>Data Requirements</h3>
    <p>The CSV file must include all the original features used during model training:</p>
    <ul>
        <li><strong>Numerical:</strong> Age, DailyRate, DistanceFromHome, Education, EmployeeCount, etc.</li>
        <li><strong>Categorical:</strong> Attrition, BusinessTravel, Department, EducationField, Gender, etc.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Add model information
    if saved_objects:
        st.markdown("### Model Information")
        st.write(f"**Model Type:** Logistic Regression")
        st.write(f"**Numerical Features:** {len(numerical_cols)}")
        st.write(f"**Categorical Features:** {len(categorical_cols)}")
        st.write(f"**Total Features after Encoding:** {len(numerical_cols) + len(encoded_cols)}")

