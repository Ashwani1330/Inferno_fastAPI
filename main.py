from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # Add this import
from fastapi.templating import Jinja2Templates
import os
from tasks.background_tasks import start_background_tasks

# Import all routers
from api.routes.base import router as base_router
from api.routes.performance import router as performance_router
from api.routes.analytics import router as analytics_router

from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import asyncio as aioredis
import os

# Create FastAPI app
app = FastAPI(
    title="Inferno VR Fire-Safety Training API",
    description="API for Immersive Navigation for Fire Emergency Response & Neutralization Operations research",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include all routers
app.include_router(base_router)
app.include_router(performance_router, prefix="/api")
app.include_router(analytics_router, prefix="/api")

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
    FastAPICache.init(RedisBackend(redis), prefix="inferno-dashboard-cache")

# Start background tasks on application startup
@app.on_event("startup")
async def startup_event():
    # ...existing startup code...
    
    # Start background tasks
    start_background_tasks()

'''
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

# Add these imports at the top of your file
from fastapi.responses import HTMLResponse
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import json
# ...existing imports...

# Add this import at the top
from fastapi.responses import RedirectResponse

# Add this route
@app.get("/analytics")
async def redirect_to_dashboard():
    """Redirects to the analytics dashboard."""
    return RedirectResponse(url="/dashboard")

# Add this endpoint after your existing routes
@app.get("/dashboard", response_class=HTMLResponse)
async def analytics_dashboard():
    """Serves a comprehensive analytics dashboard for research purposes."""
    try:
        # Fetch all performance data
        performances = await mongo_service.get_performances()
        
        if not performances:
            return "<h1>No data available yet</h1>"
        
        # Convert to DataFrame and anonymize
        df = pd.DataFrame(performances)
        # Anonymize by removing emails or replacing with generic IDs
        if 'email' in df.columns:
            df['user_id'] = [f"User_{i+1}" for i in range(len(df))]
            df = df.drop('email', axis=1)
        
        # Clean data
        event_columns = ['timeToFindExtinguisher', 'timeToExtinguishFire', 'timeToTriggerAlarm', 'timeToFindExit']
        for col in event_columns:
            df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)
        
        # Process timestamps 
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
        
        # Add age groups for deeper analysis
        if 'age' in df.columns:
            df['numeric_age'] = df['age'].apply(lambda x: parse_age(x) if isinstance(x, str) else x)
            df['age_group'] = pd.cut(df['numeric_age'], 
                                     bins=[0, 18, 30, 45, 60, 100], 
                                     labels=['Under 18', '18-30', '31-45', '46-60', 'Over 60'])
        
        # Generate HTML dashboard
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Inferno VR Fire Safety Research Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                h1, h2, h3 { color: #333; }
                .metrics { display: flex; flex-wrap: wrap; justify-content: space-between; margin-bottom: 20px; }
                .metric-card { background-color: #fff; border-radius: 8px; box-shadow: 0 0 5px rgba(0,0,0,0.1); padding: 15px; width: 23%; margin-bottom: 15px; }
                .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
                .metric-label { font-size: 14px; color: #7f8c8d; }
                .chart-container { margin-bottom: 30px; background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 0 5px rgba(0,0,0,0.1); }
                .chart-row { display: flex; flex-wrap: wrap; justify-content: space-between; }
                .chart { width: 48%; margin-bottom: 20px; }
                @media (max-width: 768px) {
                    .metric-card { width: 48%; }
                    .chart { width: 100%; }
                }
                table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
                th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                tr:hover { background-color: #f5f5f5; }
                .tab-container { margin-bottom: 20px; }
                .tab-buttons { overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }
                .tab-button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 17px; }
                .tab-button:hover { background-color: #ddd; }
                .tab-button.active { background-color: #ccc; }
                .tab-content { display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }
            </style>
            <script>
                function openTab(evt, tabName) {
                    var i, tabcontent, tabbuttons;
                    tabcontent = document.getElementsByClassName("tab-content");
                    for (i = 0; i < tabcontent.length; i++) {
                        tabcontent[i].style.display = "none";
                    }
                    tabbuttons = document.getElementsByClassName("tab-button");
                    for (i = 0; i < tabbuttons.length; i++) {
                        tabbuttons[i].className = tabbuttons[i].className.replace(" active", "");
                    }
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                }
            </script>
        </head>
        <body>
            <div class="container">
                <h1>Inferno VR Fire Safety Research Dashboard</h1>
                <p>Anonymized data analysis for research on Immersive Navigation for Fire Emergency Response & Neutralization Operations</p>
                <div class="tab-container">
                    <div class="tab-buttons">
                        <button class="tab-button active" onclick="openTab(event, 'overview')">Overview</button>
                        <button class="tab-button" onclick="openTab(event, 'demographics')">Demographics</button>
                        <button class="tab-button" onclick="openTab(event, 'performance')">Performance Analysis</button>
                        <button class="tab-button" onclick="openTab(event, 'correlations')">Correlations</button>
                        <button class="tab-button" onclick="openTab(event, 'time-analysis')">Time Analysis</button>
                    </div>
                    
                    <div id="overview" class="tab-content" style="display: block;">
        """
        
        # Overview Tab Content
        # Key metrics
        html += """
                        <h2>Key Metrics</h2>
                        <div class="metrics">
        """
        
        # Total participants
        participant_count = len(df)
        html += f"""
                            <div class="metric-card">
                                <div class="metric-value">{participant_count}</div>
                                <div class="metric-label">Total Participants</div>
                            </div>
        """
        
        # Average performance score
        avg_score = round(df['performanceScore'].mean(), 2)
        html += f"""
                            <div class="metric-card">
                                <div class="metric-value">{avg_score}</div>
                                <div class="metric-label">Average Performance Score</div>
                            </div>
        """
        
        # Average evacuation time (sum of all times)
        avg_evacuation_time = round(df[event_columns].sum(axis=1).mean(), 2)
        html += f"""
                            <div class="metric-card">
                                <div class="metric-value">{avg_evacuation_time} s</div>
                                <div class="metric-label">Average Evacuation Time</div>
                            </div>
        """
        
        # Success rate (positive score)
        success_rate = round((df['performanceScore'] > 0).mean() * 100, 2)
        html += f"""
                            <div class="metric-card">
                                <div class="metric-value">{success_rate}%</div>
                                <div class="metric-label">Success Rate</div>
                            </div>
        """
        
        html += """
                        </div>
        """
        
        # Performance distribution chart
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(df['performanceScore'], kde=True)
        plt.title('Distribution of Performance Scores')
        plt.xlabel('Performance Score')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        html += f"""
                        <div class="chart-container">
                            <h3>Performance Score Distribution</h3>
                            <img src="data:image/png;base64,{img_str}" width="100%">
                        </div>
        """
        
        # Time metrics comparison
        fig = plt.figure(figsize=(10, 6))
        time_means = df[event_columns].mean().sort_values()
        sns.barplot(x=time_means.values, y=time_means.index)
        plt.title('Average Time by Task')
        plt.xlabel('Time (seconds)')
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        html += f"""
                        <div class="chart-container">
                            <h3>Average Time by Task</h3>
                            <img src="data:image/png;base64,{img_str}" width="100%">
                        </div>
        """
        
        # Recent participation trend
        if 'date' in df.columns:
            daily_counts = df.groupby('date').size()
            fig = plt.figure(figsize=(10, 6))
            plt.plot(daily_counts.index, daily_counts.values, marker='o')
            plt.title('Participation Trend')
            plt.xlabel('Date')
            plt.ylabel('Number of Participants')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            html += f"""
                            <div class="chart-container">
                                <h3>Participation Trend</h3>
                                <img src="data:image/png;base64,{img_str}" width="100%">
                            </div>
            """
        
        # Latest records
        html += """
                        <h3>Latest Records</h3>
                        <table>
                            <tr>
                                <th>User ID</th>
                                <th>Age</th>
                                <th>Scene</th>
                                <th>Difficulty</th>
                                <th>Score</th>
                                <th>Date</th>
                            </tr>
        """
        
        if 'timestamp' in df.columns:
            latest_records = df.sort_values('timestamp', ascending=False).head(10)
            for _, row in latest_records.iterrows():
                date_str = row['timestamp'].strftime('%Y-%m-%d') if 'timestamp' in row else 'N/A'
                html += f"""
                            <tr>
                                <td>{row.get('user_id', 'Anonymous')}</td>
                                <td>{row.get('age', 'N/A')}</td>
                                <td>{row.get('sceneType', 'N/A')}</td>
                                <td>{row.get('difficulty', 'N/A')}</td>
                                <td>{row.get('performanceScore', 'N/A'):.2f}</td>
                                <td>{date_str}</td>
                            </tr>
                """
        
        html += """
                        </table>
                    </div>
        """
        
        # Demographics Tab
        html += """
                    <div id="demographics" class="tab-content">
                        <h2>Demographic Analysis</h2>
        """
        
        # Age distribution
        if 'numeric_age' in df.columns:
            fig = plt.figure(figsize=(10, 6))
            sns.histplot(df['numeric_age'], kde=True, bins=20)
            plt.title('Age Distribution of Participants')
            plt.xlabel('Age')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            html += f"""
                        <div class="chart-container">
                            <h3>Age Distribution</h3>
                            <img src="data:image/png;base64,{img_str}" width="100%">
                        </div>
            """
        
        # Performance by age group
        if 'age_group' in df.columns:
            fig = plt.figure(figsize=(10, 6))
            sns.boxplot(x='age_group', y='performanceScore', data=df)
            plt.title('Performance Score by Age Group')
            plt.xlabel('Age Group')
            plt.ylabel('Performance Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            html += f"""
                        <div class="chart-container">
                            <h3>Performance by Age Group</h3>
                            <img src="data:image/png;base64,{img_str}" width="100%">
                        </div>
            """
            
            # Time metrics by age group
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Time Metrics by Age Group', fontsize=16)
            
            for i, col in enumerate(event_columns):
                row, col_idx = divmod(i, 2)
                sns.boxplot(x='age_group', y=col, data=df, ax=axs[row, col_idx])
                axs[row, col_idx].set_title(col)
                axs[row, col_idx].set_xlabel('Age Group')
                axs[row, col_idx].set_ylabel('Time (seconds)')
                axs[row, col_idx].tick_params(axis='x', rotation=45)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            html += f"""
                        <div class="chart-container">
                            <h3>Time Metrics by Age Group</h3>
                            <img src="data:image/png;base64,{img_str}" width="100%">
                        </div>
            """
        
        # Scene type distribution
        if 'sceneType' in df.columns:
            scene_counts = df['sceneType'].value_counts()
            fig = plt.figure(figsize=(10, 6))
            plt.pie(scene_counts, labels=scene_counts.index, autopct='%1.1f%%', startangle=90)
            plt.title('Distribution by Scene Type')
            plt.axis('equal')
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            html += f"""
                        <div class="chart-container">
                            <h3>Scene Type Distribution</h3>
                            <img src="data:image/png;base64,{img_str}" width="100%">
                        </div>
            """
        
        html += """
                    </div>
        """
        
        # Performance Analysis Tab
        html += """
                    <div id="performance" class="tab-content">
                        <h2>Performance Analysis</h2>
        """
        
        # Performance by scene type
        if 'sceneType' in df.columns:
            fig = plt.figure(figsize=(10, 6))
            sns.boxplot(x='sceneType', y='performanceScore', data=df)
            plt.title('Performance Score by Scene Type')
            plt.xlabel('Scene Type')
            plt.ylabel('Performance Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            html += f"""
                        <div class="chart-container">
                            <h3>Performance by Scene Type</h3>
                            <img src="data:image/png;base64,{img_str}" width="100%">
                        </div>
            """
        
        # Performance by difficulty
        if 'difficulty' in df.columns:
            fig = plt.figure(figsize=(10, 6))
            sns.boxplot(x='difficulty', y='performanceScore', data=df)
            plt.title('Performance Score by Difficulty Level')
            plt.xlabel('Difficulty')
            plt.ylabel('Performance Score')
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            html += f"""
                        <div class="chart-container">
                            <h3>Performance by Difficulty Level</h3>
                            <img src="data:image/png;base64,{img_str}" width="100%">
                        </div>
            """
        
        # Performance comparison by tasks
        fig = plt.figure(figsize=(10, 8))
        task_performance = pd.DataFrame()
        for col in event_columns:
            task_performance[col] = df[col] / df[col].max()  # Normalize for comparison
        
        # Melt the dataframe for easier plotting
        task_performance_melted = pd.melt(task_performance.reset_index(), 
                                         id_vars=['index'],
                                         value_vars=event_columns,
                                         var_name='Task',
                                         value_name='Normalized Time')
        
        sns.boxplot(x='Task', y='Normalized Time', data=task_performance_melted)
        plt.title('Normalized Task Performance Comparison')
        plt.xlabel('Task')
        plt.ylabel('Normalized Time (lower is better)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        html += f"""
                        <div class="chart-container">
                            <h3>Task Performance Comparison</h3>
                            <img src="data:image/png;base64,{img_str}" width="100%">
                        </div>
        """
        
        html += """
                    </div>
        """
        
        # Correlations Tab
        html += """
                    <div id="correlations" class="tab-content">
                        <h2>Correlation Analysis</h2>
        """
        
        # Correlation heatmap
        fig = plt.figure(figsize=(12, 10))
        correlation_cols = ['numeric_age'] + event_columns + ['performanceScore'] if 'numeric_age' in df.columns else event_columns + ['performanceScore']
        correlation_df = df[correlation_cols].copy()
        
        # Calculate correlation matrix
        corr_matrix = correlation_df.corr()
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        html += f"""
                        <div class="chart-container">
                            <h3>Correlation Heatmap</h3>
                            <img src="data:image/png;base64,{img_str}" width="100%">
                        </div>
        """
        
        # Scatter plots for important correlations
        if 'numeric_age' in df.columns:
            # Age vs Performance
            fig = plt.figure(figsize=(10, 6))
            sns.scatterplot(x='numeric_age', y='performanceScore', data=df)
            plt.title('Age vs Performance Score')
            plt.xlabel('Age')
            plt.ylabel('Performance Score')
            plt.grid(True, alpha=0.3)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            html += f"""
                            <div class="chart-container">
                                <h3>Age vs Performance</h3>
                                <img src="data:image/png;base64,{img_str}" width="100%">
                            </div>
            """
        
        # Pairplot of key metrics
        fig = plt.figure(figsize=(12, 10))
        g = sns.pairplot(df[event_columns], diag_kind='kde')
        plt.suptitle('Relationships Between Time Metrics', y=1.02)
        
        buf = io.BytesIO()
        g.fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(g.fig)
        
        html += f"""
                        <div class="chart-container">
                            <h3>Relationships Between Time Metrics</h3>
                            <img src="data:image/png;base64,{img_str}" width="100%">
                        </div>
        """
        
        html += """
                    </div>
        """
        
        # Time Analysis Tab
        html += """
                    <div id="time-analysis" class="tab-content">
                        <h2>Time Series Analysis</h2>
        """
        
        if 'timestamp' in df.columns:
            # Performance over time
            df_sorted = df.sort_values('timestamp')
            fig = plt.figure(figsize=(12, 6))
            plt.plot(df_sorted['timestamp'], df_sorted['performanceScore'], marker='o', alpha=0.7)
            plt.title('Performance Scores Over Time')
            plt.xlabel('Time')
            plt.ylabel('Performance Score')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            html += f"""
                        <div class="chart-container">
                            <h3>Performance Trend</h3>
                            <img src="data:image/png;base64,{img_str}" width="100%">
                        </div>
            """
            
            # Average performance by day
            df['day'] = df['timestamp'].dt.date
            daily_avg = df.groupby('day')['performanceScore'].mean()
            
            fig = plt.figure(figsize=(12, 6))
            plt.plot(daily_avg.index, daily_avg.values, marker='o', linestyle='-')
            plt.title('Average Daily Performance')
            plt.xlabel('Date')
            plt.ylabel('Average Performance Score')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            html += f"""
                        <div class="chart-container">
                            <h3>Average Daily Performance</h3>
                            <img src="data:image/png;base64,{img_str}" width="100%">
                        </div>
            """
            
            # Task time improvement over time
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Task Times Trend Over Time', fontsize=16)
            
            for i, col in enumerate(event_columns):
                row, col_idx = divmod(i, 2)
                # Create a rolling average to see trends more clearly
                df_sorted[f'{col}_rolling'] = df_sorted[col].rolling(window=5, min_periods=1).mean()
                
                axs[row, col_idx].plot(df_sorted['timestamp'], df_sorted[col], 'o', alpha=0.3)
                axs[row, col_idx].plot(df_sorted['timestamp'], df_sorted[f'{col}_rolling'], '-', color='red')
                axs[row, col_idx].set_title(col)
                axs[row, col_idx].set_xlabel('Time')
                axs[row, col_idx].set_ylabel('Time (seconds)')
                axs[row, col_idx].tick_params(axis='x', rotation=45)
                axs[row, col_idx].grid(True, alpha=0.3)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            html += f"""
                        <div class="chart-container">
                            <h3>Task Times Trend</h3>
                            <img src="data:image/png;base64,{img_str}" width="100%">
                        </div>
            """
        
        html += """
                    </div>
                </div>
                
                <div style="margin-top: 30px;">
                    <h2>Research Insights</h2>
                    <p>This dashboard provides anonymized data from participants in the Inferno VR Fire-Safety Training program. The data presented here forms the basis for research on Immersive Navigation for Fire Emergency Response & Neutralization Operations.</p>
                    <p>Key insights:</p>
                    <ul>
                        <li>The average performance score across all participants is {avg_score:.2f}</li>
                        <li>The relationship between age and performance shows {df['numeric_age'].corr(df['performanceScore']):.2f} correlation</li>
                        <li>Success rate (positive performance score): {success_rate:.2f}%</li>
                        <li>Total number of training sessions: {participant_count}</li>
                    </ul>
                    <p>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        return f"<h1>Error generating dashboard</h1><p>{str(e)}</p>"

'''