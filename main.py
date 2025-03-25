from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from models.performance import PerformanceInput, PerformanceOutput
from services.mongo_service import MongoService
from services.email_service import EmailService
from services.analysis_service import AnalysisService
from utils.score_calculator import calculate_evacuation_efficiency_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from datetime import datetime

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
mongo_service = MongoService()
email_service = EmailService()
analysis_service = AnalysisService()

# Helper functions
def parse_age(age_str):
    if "-" in age_str:
        age_parts = age_str.split("-")
        if len(age_parts) == 2:
            return (int(age_parts[0]) + int(age_parts[1])) / 2
    return int(age_str)

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to the Inferno VR Fire-Safety Training API"}

@app.post("/api/performance", response_model=PerformanceOutput)
async def create_performance(performance: PerformanceInput):
    try:
        numeric_age = parse_age(performance.age)
        
        times = {
            "timeToFindExtinguisher": performance.timeToFindExtinguisher,
            "timeToExtinguishFire": performance.timeToExtinguishFire,
            "timeToTriggerAlarm": performance.timeToTriggerAlarm,
            "timeToFindExit": performance.timeToFindExit
        }
        
        performance_score = calculate_evacuation_efficiency_score(numeric_age, times)
        
        performance_data = {
            "email": performance.email,
            "age": performance.age,
            "sceneType": performance.sceneType,
            "difficulty": performance.difficulty,
            "timeToFindExtinguisher": performance.timeToFindExtinguisher,
            "timeToExtinguishFire": performance.timeToExtinguishFire,
            "timeToTriggerAlarm": performance.timeToTriggerAlarm,
            "timeToFindExit": performance.timeToFindExit,
            "performanceScore": performance_score,
            "timestamp": datetime.now()
        }
        
        await mongo_service.insert_performance(performance_data)
        
        # Generate and send report via email
        if performance.email:
            performances = await mongo_service.get_performances()
            df = pd.DataFrame(performances)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            latest_record = df.iloc[-1]
            report_html = analysis_service.generate_report(df, latest_record)
            email_service.send_email(performance.email, "Performance Report", report_html)
        
        return {"message": "Data saved successfully", "performanceScore": performance_score}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/performance")
async def get_performances():
    performances = await mongo_service.get_performances()
    
    # Convert ObjectId to string for JSON serialization
    for performance in performances:
        performance["_id"] = str(performance["_id"])
    
    return performances

@app.get("/api/performance/report")
async def get_performance_report():
    # Fetch all performance data
    performances = await mongo_service.get_performances()
    
    if not performances:
        raise HTTPException(status_code=404, detail="No performance data found")
    
    # Convert to DataFrame
    df = pd.DataFrame(performances)
    
    # Clean data
    event_columns = ['timeToFindExtinguisher', 'timeToExtinguishFire', 'timeToTriggerAlarm', 'timeToFindExit']
    score_column = 'performanceScore'
    
    for col in event_columns:
        df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)
    
    # Get latest record
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        latest_record = df.iloc[-1]
    else:
        latest_record = df.iloc[-1]
    
    # Compute percentiles
    metrics = {}
    all_metrics = event_columns + [score_column]
    
    for col in all_metrics:
        if col in df.columns:
            latest_value = latest_record[col]
            percentile = analysis_service.compute_percentile(df[col], latest_value)
            metrics[col] = {
                "value": latest_value,
                "percentile": percentile
            }
    
    # Generate graphs
    graphs_html = {}
    for col in all_metrics:
        if col in df.columns:
            graphs_html[col] = analysis_service.generate_metric_graph(col, latest_record[col], df[col])
    
    # Create HTML report
    report_html = "<h1>Latest Performance Report</h1>"
    report_html += f"<p>Date: {datetime.now().strftime('%Y-%m-%d')}</p>"
    
    if 'timestamp' in latest_record:
        report_html += f"<p>Record Timestamp: {latest_record['timestamp']}</p>"
    
    report_html += "<h2>Performance Metrics</h2>"
    report_html += "<table border='1' cellpadding='5' cellspacing='0'>"
    report_html += "<tr><th>Metric</th><th>Your Value</th><th>Percentile Rank</th></tr>"
    
    for metric, data in metrics.items():
        report_html += f"<tr><td>{metric}</td><td>{data['value']}</td><td>{data['percentile']:.2f}%</td></tr>"
    
    report_html += "</table>"
    
    report_html += "<h2>Visualizations</h2>"
    for metric, img_html in graphs_html.items():
        report_html += f"<h3>{metric}</h3>"
        report_html += img_html
    
    return {"report_html": report_html}

# Helper function to clean data for JSON serialization
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (np.float64, np.float32, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, datetime.date):
        return obj.strftime('%Y-%m-%d')
    else:
        return obj

@app.get("/api/performance/analysis")
async def get_performance_analysis():
    # Fetch all performance data
    performances = await mongo_service.get_performances()
    
    if not performances:
        raise HTTPException(status_code=404, detail="No performance data found")
    
    # Convert to DataFrame
    df = pd.DataFrame(performances)
    
    # Clean data
    event_columns = ['timeToFindExtinguisher', 'timeToExtinguishFire', 'timeToTriggerAlarm', 'timeToFindExit']
    for col in event_columns:
        df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)
    
    # Convert timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')  # Convert to string for JSON
    
    # Replace NaN, Infinity, and -Infinity with None
    df = df.replace([np.nan, np.inf, -np.inf], None)
    
    # Basic statistics
    stats = df[event_columns + ['performanceScore']].describe().to_dict()
    
    # Correlation analysis
    try:
        df['numeric_age'] = df['age'].apply(lambda x: parse_age(x) if isinstance(x, str) else x)
        corr = df[['numeric_age'] + event_columns + ['performanceScore']].corr().to_dict()
    except Exception as e:
        corr = {"error": str(e)}
    
    # Group by analysis
    difficulty_analysis = {}
    scene_analysis = {}
    if 'difficulty' in df.columns:
        difficulty_analysis = df.groupby('difficulty')['performanceScore'].agg(['mean', 'median', 'std']).to_dict()
    if 'sceneType' in df.columns:
        scene_analysis = df.groupby('sceneType')['performanceScore'].agg(['mean', 'median', 'std']).to_dict()
    
    # Predictive modeling
    model_results = {}
    try:
        features = ['numeric_age'] + event_columns
        model_df = df.dropna(subset=features + ['performanceScore'])
        
        if len(model_df) > 10:  # Only run if we have enough data
            X = model_df[features]
            y = model_df['performanceScore']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            
            y_pred = lr.predict(X_test)
            
            model_results = {
                "mse": mean_squared_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "coefficients": dict(zip(features, lr.coef_))
            }
    except Exception as e:
        model_results = {"error": str(e)}
    
    # Time series analysis
    time_series = {}
    if 'timestamp' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            daily = df.groupby('date')['performanceScore'].mean()
            # Convert date indices to strings for JSON serialization
            time_series = {str(date): value for date, value in daily.items()}
        except Exception as e:
            time_series = {"error": str(e)}
    
    result = {
        "basic_stats": stats,
        "correlation": corr,
        "difficulty_analysis": difficulty_analysis,
        "scene_analysis": scene_analysis,
        "model_results": model_results,
        "time_series": time_series
    }
    
    # Clean the entire response to ensure JSON compatibility
    cleaned_result = clean_for_json(result)
    return cleaned_result