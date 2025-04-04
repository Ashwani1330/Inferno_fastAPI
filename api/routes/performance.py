from fastapi import APIRouter, HTTPException, Response
import pandas as pd
import numpy as np
import asyncio
import io
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime

from models.performance import PerformanceInput, PerformanceOutput
from services.mongo_service import MongoService
from services.email_service import EmailService
from services.analysis_service import AnalysisService
from utils.score_calculator import calculate_evacuation_efficiency_score
from utils.helpers import parse_age, clean_for_json
from tasks.background_tasks import trigger_analytics_update

router = APIRouter()

# Initialize services
mongo_service = MongoService()
email_service = EmailService()
analysis_service = AnalysisService()

@router.post("/performance", response_model=PerformanceOutput)
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
        
        # Trigger analytics update in background
        asyncio.create_task(trigger_analytics_update())
        
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

@router.get("/performance")
async def get_performances():
    performances = await mongo_service.get_performances()
    
    # Convert ObjectId to string for JSON serialization
    for performance in performances:
        performance["_id"] = str(performance["_id"])
    
    return performances

@router.get("/performance/report")
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

@router.get("/performance/analysis")
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

@router.get("/export/csv")
async def export_csv():
    performances = await mongo_service.get_performances()
    df = pd.DataFrame(performances)
    csv_data = df.to_csv(index=False)
    return Response(content=csv_data, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=performance_data.csv"})

@router.get("/export/excel")
async def export_excel():
    performances = await mongo_service.get_performances()
    df = pd.DataFrame(performances)
    excel_buffer = io.BytesIO()
    try:
        # Try using openpyxl engine
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name='Performance Data')
    except ModuleNotFoundError as e1:
        try:
            # Fallback to xlsxwriter engine if openpyxl is not available
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name='Performance Data')
        except ModuleNotFoundError as e2:
            # Neither openpyxl nor xlsxwriter is installed
            error_msg = (
                "Excel export libraries are not installed. Please install required packages using:\n"
                "pip install openpyxl xlsxwriter\n\n"
                "Alternatively, you can use CSV export which doesn't require additional packages."
            )
            raise HTTPException(status_code=500, detail=error_msg) from e2
    
    excel_buffer.seek(0)
    return Response(content=excel_buffer.read(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=performance_data.xlsx"})