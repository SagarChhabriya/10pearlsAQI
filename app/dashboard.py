"""Streamlit dashboard for AQI predictions."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import os
from pathlib import Path
import sys
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AQI Predictor Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get API URL from environment or use default
API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Add requests to imports
import requests


@st.cache_data(ttl=300)
def get_predictions(forecast_days: int = 3, latitude: float = None, longitude: float = None):
    """Get predictions from FastAPI service."""
    try:
        url = f"{API_URL}/predict"
        payload = {
            "forecast_days": forecast_days
        }
        if latitude and longitude:
            payload["latitude"] = latitude
            payload["longitude"] = longitude
        
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling prediction API: {str(e)}")
        return None


@st.cache_data(ttl=300)
def get_available_models():
    """Get list of available models from API."""
    try:
        url = f"{API_URL}/models"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting models: {str(e)}")
        return None


def get_aqi_category(aqi: float) -> tuple:
    """Get AQI category and color based on AQI value."""
    if aqi <= 50:
        return "Good", "green", ""
    elif aqi <= 100:
        return "Moderate", "yellow", ""
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "orange", ""
    elif aqi <= 200:
        return "Unhealthy", "red", ""
    elif aqi <= 300:
        return "Very Unhealthy", "purple", ""
    else:
        return "Hazardous", "maroon", ""




def main():
    """Main dashboard function."""
    st.markdown('<h1 class="main-header">AQI Predictor Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.text(f"City: {config['city']['name']}")
        st.text(f"Location: {config['city']['latitude']:.4f}, {config['city']['longitude']:.4f}")
        
        forecast_days = st.slider("Forecast Days", 1, 7, 3)
        
        # Use configured city location
        latitude = config['city']['latitude']
        longitude = config['city']['longitude']
        
        st.header("About")
        st.info("""
        This dashboard provides real-time AQI predictions 
        for the next 3 days using machine learning models.
        """)
        
        st.header("API Status")
        try:
            health_response = requests.get(f"{API_URL}/health", timeout=5)
            if health_response.status_code == 200:
                st.success("API Connected")
            else:
                st.error("API Unavailable")
        except:
            st.error("API Unavailable")
    
    # Check API connection
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code != 200:
            st.error("Prediction API is not available. Please ensure the FastAPI service is running.")
            st.info("Start the API with: `uvicorn api.main:app --host 0.0.0.0 --port 8000`")
            return
    except:
        st.error("Cannot connect to Prediction API. Please ensure the FastAPI service is running.")
        st.info("Start the API with: `uvicorn api.main:app --host 0.0.0.0 --port 8000`")
        return
    
    # Get model info
    models_info = get_available_models()
    if models_info and isinstance(models_info, dict) and models_info.get('models'):
        st.header("Model Information")
        
        # Find and highlight the best model (lowest RMSE, excluding invalid models)
        best_model = None
        best_rmse = float('inf')
        for model in models_info['models']:
            if isinstance(model, dict):
                rmse = model.get('metrics', {}).get('rmse', float('inf'))
                # Only consider models with valid RMSE (> 0)
                if isinstance(rmse, (int, float)) and rmse > 0 and rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
        
        # Display best model prominently
        if best_model:
            st.success(f"**Best Model**: {best_model.get('name', 'Unknown')} (RMSE: {best_model.get('metrics', {}).get('rmse', 'N/A'):.2f})")
        
        # Display best model metrics in columns (not latest, but best)
        col1, col2, col3 = st.columns(3)
        with col1:
            if best_model:
                st.metric("Best Model", best_model.get('name', 'Unknown'))
            else:
                st.metric("Best Model", "N/A")
        with col2:
            if best_model and best_model.get('metrics'):
                r2 = best_model['metrics'].get('r2', 0)
                st.metric("R² Score", f"{r2:.3f}" if isinstance(r2, (int, float)) else "N/A")
            else:
                st.metric("R² Score", "N/A")
        with col3:
            if best_model and best_model.get('metrics'):
                rmse = best_model['metrics'].get('rmse', 0)
                st.metric("RMSE", f"{rmse:.2f}" if isinstance(rmse, (int, float)) else "N/A")
            else:
                st.metric("RMSE", "N/A")
    elif models_info:
        st.header("Model Information")
        st.warning("Model information is not available in the expected format.")
    
    # Get predictions
    st.header("Current Air Quality and Forecast")
    st.info("Click the button below to fetch real-time AQI data and generate predictions for the configured city.")
    
    if st.button("Get Predictions", type="primary"):
        with st.spinner("Fetching predictions..."):
            predictions_data = get_predictions(forecast_days=forecast_days, latitude=latitude, longitude=longitude)
            
            if predictions_data is None:
                st.error("Could not fetch predictions. Please check API connection.")
                return
            
            # Display model information used for prediction
            if predictions_data.get('model_name'):
                model_name = predictions_data.get('model_name', 'Unknown')
                model_metrics = predictions_data.get('model_metrics', {})
                rmse = model_metrics.get('rmse', 'N/A')
                r2 = model_metrics.get('r2', 'N/A')
                
                # Format metrics properly
                rmse_str = f"{rmse:.2f}" if isinstance(rmse, (int, float)) else str(rmse)
                r2_str = f"{r2:.3f}" if isinstance(r2, (int, float)) else str(r2)
                
                st.success(f"**Model Used for Prediction**: {model_name} | **RMSE**: {rmse_str} | **R²**: {r2_str}")
            
            # Display current AQI
            if predictions_data.get('current_aqi'):
                current_aqi = predictions_data['current_aqi']
                category, color, _ = get_aqi_category(current_aqi)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current AQI", f"{current_aqi:.0f}")
                with col2:
                    st.markdown(f"### {category}")
                with col3:
                    if current_aqi > config['dashboard']['alert_thresholds']['unhealthy']:
                        st.error("Alert: Unhealthy air quality detected!")
            
            # Display predictions
            predictions = predictions_data.get('predictions', [])
            if predictions:
                pred_df = pd.DataFrame(predictions)
                
                # Create visualization
                fig = go.Figure()
                
                for _, row in pred_df.iterrows():
                    aqi = row.get('predicted_aqi')
                    if aqi is not None:
                        category, color, _ = get_aqi_category(aqi)
                        
                        fig.add_trace(go.Bar(
                            x=[row['date']],
                            y=[aqi],
                            name=f"Day {row['day']}",
                            marker_color=color,
                            text=f"{aqi:.0f}",
                            textposition='outside',
                            hovertemplate=f"<b>{row['date']}</b><br>AQI: {aqi:.0f}<br>Category: {category}<extra></extra>"
                        ))
                
                # Set Y-axis range to show full AQI scale (0-500 EPA standard)
                max_aqi = max(pred_df['predicted_aqi'].max() if not pred_df.empty else 100, 100)
                yaxis_max = max(500, max_aqi * 1.2)  # Show up to 500, or 20% above max value
                
                fig.update_layout(
                    title=f"AQI Forecast for Next {forecast_days} Days",
                    xaxis_title="Date",
                    yaxis_title="AQI",
                    yaxis=dict(range=[0, yaxis_max], dtick=50),  # Fixed range with 50-unit ticks
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display predictions table
                st.subheader("Forecast Details")
                display_df = pred_df.copy()
                display_df['Predicted AQI'] = display_df['predicted_aqi'].apply(lambda x: f"{x:.0f}" if x else "N/A")
                display_df = display_df[['date', 'Predicted AQI', 'category']]
                display_df.columns = ['Date', 'Predicted AQI', 'Category']
                st.dataframe(display_df, use_container_width=True)
                
                # Alerts
                st.subheader("Alerts")
                alerts = []
                for _, row in pred_df.iterrows():
                    aqi = row.get('predicted_aqi')
                    if aqi and aqi > config['dashboard']['alert_thresholds']['unhealthy']:
                        alerts.append(f"{row['date']}: Unhealthy AQI predicted ({aqi:.0f})")
                
                if alerts:
                    for alert in alerts:
                        st.warning(alert)
                else:
                    st.success("No alerts - air quality is expected to be within acceptable limits.")
    
    # Additional information
    with st.expander("About AQI Categories"):
        st.markdown("""
        - **Good (0-50)**: Air quality is satisfactory.
        - **Moderate (51-100)**: Acceptable for most people.
        - **Unhealthy for Sensitive Groups (101-150)**: Sensitive groups may experience health effects.
        - **Unhealthy (151-200)**: Everyone may begin to experience health effects.
        - **Very Unhealthy (201-300)**: Health alert - everyone may experience serious health effects.
        - **Hazardous (301+)**: Health warning - entire population likely affected.
        """)


if __name__ == "__main__":
    main()
