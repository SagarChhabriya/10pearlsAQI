# 10pearlsAQI

https://10pearls-aqi-predictor.streamlit.app/

An end-to-end Air Quality Index (AQI) prediction system using a 100% serverless stack. This project predicts AQI for the next 3 days using machine learning models, automated pipelines, and an interactive dashboard.

Portal: https://shine.10pearls.com/candidate/submissions

![alt text](assets/image.png)

## Quick Start

### Prerequisites
- Python 3.10 or higher
- MongoDB Atlas account (for feature storage)
- API keys for:
  - AQICN (https://aqicn.org/api/)
  - OpenWeather (https://openweathermap.org/api)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd 10pearlsAQI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   - Copy `config/env.example` to `.env` (in project root)
   - Add your API keys and MongoDB URI:
     ```
     AQICN_TOKEN=your_token_here
     OPENWEATHER_TOKEN=your_token_here
     MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/database
     FASTAPI_URL=http://localhost:8000
     ```

5. **Update configuration**
   - Edit `config/config.yaml` to set your city coordinates and preferences

### Project Structure

```
10pearlsAQI/
├── api/                      # FastAPI service
│   ├── __init__.py
│   ├── main.py              # API endpoints
│   ├── requirements.txt     # API dependencies
│   └── Dockerfile           # Docker configuration
├── app/                      # Streamlit dashboard
│   ├── __init__.py
│   └── dashboard.py         # Dashboard application
├── pipelines/                # Data and ML pipelines
│   ├── __init__.py
│   ├── data_fetcher.py      # API data fetching
│   ├── data_cleaning.py     # Data cleaning
│   ├── feature_engineering.py  # Feature creation
│   ├── feature_pipeline.py  # Main feature pipeline
│   ├── training_pipeline.py # Model training
│   ├── backfill.py          # Historical data backfill
│   ├── mongodb_store.py     # MongoDB storage
│   ├── aqi_calculator.py    # EPA AQI calculation
│   └── utils.py             # Utility functions
├── config/                   # Configuration files
│   ├── __init__.py
│   ├── config.yaml          # Main configuration
│   ├── settings.py          # Settings loader
│   └── env.example          # Environment template
├── scripts/                   # Utility scripts
│   ├── README.md            # Scripts documentation
│   ├── quick_start.py       # Setup verification
│   ├── setup_cloud.py       # Cloud setup wizard
│   ├── run_dashboard.py     # Dashboard launcher
│   ├── test_mongodb.py      # MongoDB connection test
│   ├── clear_features.py    # Clear MongoDB features
│   ├── export_to_csv.py     # Export data to CSV
│   ├── check_data_quality.py # Data quality checks
│   ├── check_overfitting.py  # Model overfitting analysis
│   └── run_optimized_backfill.py  # Optimized backfill
├── docs/                      # Documentation
│   ├── README.md            # Documentation index
│   ├── SETUP_GUIDE.md       # Complete setup guide
│   ├── EXECUTION_ORDER.md   # Step-by-step execution guide
│   ├── TECHNICAL_DOCS.md    # Technical documentation
│   ├── COMMIT_GUIDE.md      # Git commit guidelines
│   └── SUPERVISOR_REPORT.md # Project report for supervisors
├── assets/                    # Static assets
│   ├── image.png            # Project image
│   ├── AQI_predict-1.pdf    # Project specification
│   └── discord_chat.md      # Meeting notes
├── data/                      # Data storage (local fallback)
│   ├── features/            # Feature files (CSV)
│   └── raw/                 # Raw data (CSV)
├── models/                    # Trained models (local backup)
│   ├── random_forest/       # Random Forest model
│   ├── xgboost/             # XGBoost model
│   └── ensemble/            # Ensemble model
├── notebooks/                 # Jupyter notebooks for EDA
│   └── exploring_data.ipynb # Exploratory data analysis
├── tests/                     # Unit tests
├── .github/workflows/         # CI/CD pipelines
│   ├── feature_pipeline.yml  # Hourly feature pipeline
│   └── training_pipeline.yml # Daily training pipeline
├── .streamlit/                # Streamlit configuration
│   └── config.toml          # Streamlit settings
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── CONTRIBUTING.md            # Contribution guidelines
├── LICENSE                    # License file
└── .gitignore                 # Git ignore rules
```


## Usage

### 1. Run Feature Pipeline
Fetch current data and generate features:
```bash
python pipelines/feature_pipeline.py
```

### 2. Backfill Historical Data
Generate training data for past dates:
```bash
python pipelines/backfill.py --start-date 2024-01-01 --end-date 2024-01-31 --frequency daily
```

### 3. Train Models
Train ML models on historical data:
```bash
python pipelines/training_pipeline.py
```

### 4. Start FastAPI Service
Start the prediction API:
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5. Launch Dashboard
Start the Streamlit dashboard:
```bash
streamlit run app/dashboard.py
```

The dashboard will connect to the FastAPI service for predictions.

### 5. CI/CD Automation
The GitHub Actions workflows automatically:
- Run feature pipeline every hour
- Run training pipeline daily at 2 AM UTC

To enable, add your API keys as GitHub Secrets:
- `AQICN_TOKEN`
- `OPENWEATHER_TOKEN`
- `MONGODB_URI`

## Key Features

- Feature Pipeline Development
    - Fetch raw weather and pollutant data from external APIs like AQICN or OpenWeather
    - Compute features from raw data including time-based features (hour, day, month) and derived features like AQI change rate
    - Store processed features in MongoDB Atlas

- Historical Data Backfill
    - Run feature pipeline for past dates to generate training data
    - Create comprehensive dataset for model training and evaluation

- Training Pipeline Implementation
    - Fetch historical features and targets from Feature Store
    - Experiment with various ML models (Random Forest, Ridge Regression, TensorFlow/PyTorch)
    - Evaluate performance using RMSE, MAE, and R² metrics
    - Store trained models in Model Registry
- Automated CI/CD Pipeline
    - Feature pipeline runs every hour automatically
    - Training pipeline runs daily for model updates
    - Use Apache Airflow, GitHub Actions, or similar tools
- Web Application Dashboard
    - Load models and features from Feature Store
    - Compute real-time predictions for next 3 days
    - Display interactive dashboard with Streamlit/Gradio and Flask/FastAPI

- Advanced Analytics Features
    - Perform Exploratory Data Analysis (EDA) to identify trends
    - Use SHAP or LIME for feature importance explanations
    - Implement alerts for hazardous AQI levels
    - Support multiple forecasting models from statistical to deep learning


### Projects Resources

- Helpful resources and tutorials: https://drive.google.com/file/d/1HPf17hvqI6icNTjRPkPuydkV1ub_lxO5/view?usp=sharing

Due: Feb 13, 2026 at 6:20 PM
46 days remaining


### Sync up 01

Here is a summary of today's meeting session:

You will be building an AQI (Air Quality Index) Analysis bot for your city.

As part of the project, you will work with:
Data collection (using any API you like)
Data preprocessing
Feature selection
Uploading Features to MongoDB Atlas
Training your own chosen models on your dataset.
Analyzing and Reporting results
A small frontend (could be on streamlit or gradio)

And importantly a
CI/CD pipeline (using Github actions)

We will upload the task documents for this project this week. 


## Configuration

### API Setup

1. **AQICN API**
   - Sign up at https://aqicn.org/api/
   - Get your token and add to `.env` as `AQICN_TOKEN`

2. **OpenWeather API**
   - Sign up at https://openweathermap.org/api
   - Get your API key and add to `.env` as `OPENWEATHER_TOKEN`

3. **MongoDB Atlas (Required)**
   - Sign up for free tier at https://www.mongodb.com/cloud/atlas
   - Create a cluster and get connection string
   - Add to `.env` as `MONGODB_URI`
   - Features and models will be stored in MongoDB

### City Configuration

Edit `config/config.yaml` to set your city:
```yaml
city:
  name: "Your City"
  latitude: 24.8608
  longitude: 67.0104
  timezone: "America/Karachi"
```

## Model Training

The training pipeline supports multiple algorithms:
- Random Forest
- Ridge Regression
- XGBoost
- Neural Networks (TensorFlow/PyTorch)

Models are evaluated using:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)

The best model is automatically selected and saved.

## Dashboard Features

- Real-time AQI data display
- 3-day AQI forecast
- Interactive visualizations
- Alert system for hazardous levels
- Model performance metrics
- Feature importance (with SHAP/LIME)

## Development

### Running Tests
```bash
pytest tests/
```

### EDA Notebooks
Jupyter notebooks for exploratory data analysis are in the `notebooks/` directory.

## Notes

- Features are stored in MongoDB Atlas (cloud database)
- Models are stored in MongoDB for easy access by FastAPI service
- Local fallback storage available if MongoDB is unavailable
- API rate limits may apply - adjust backfill frequency accordingly
- For production, consider using cloud storage for data persistence

## Resources

- GitHub Actions Playlist: https://www.youtube.com/playlist?list=PLiO7XHcmTsleVSRaY7doSfZryYWMkMOxB
- Project Resources: https://drive.google.com/file/d/1HPf17hvqI6icNTjRPkPuydkV1ub_lxO5/view?usp=sharing

## Project Timeline

Due: Feb 13, 2026 at 6:20 PM
