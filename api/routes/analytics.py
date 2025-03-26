from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from datetime import datetime
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import asyncio as aioredis
from PIL import Image
import itertools
import asyncio

from services.mongo_service import MongoService
from utils.helpers import parse_age

router = APIRouter()

# Initialize services
mongo_service = MongoService()

# Helper function to optimize plots
def optimize_plot(fig, title, dpi=80, quality=80, format='webp', max_width=800):
    """Optimize a matplotlib figure for web display."""
    # Set title and layout
    plt.title(title)
    plt.tight_layout()
    
    # Save figure to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, 
                optimize=True, bbox_inches='tight', 
                pad_inches=0.1, transparent=False)
    buf.seek(0)
    
    # Further optimize with PIL
    img = Image.open(buf)
    
    # Resize if larger than max_width
    if img.width > max_width:
        ratio = max_width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((max_width, new_height), Image.LANCZOS)
    
    # Save as optimized format
    out_buf = io.BytesIO()
    if format == 'webp':
        img.save(out_buf, format='WEBP', quality=quality)
        mime_type = 'image/webp'
    else:
        img.save(out_buf, format='PNG', optimize=True, quality=quality)
        mime_type = 'image/png'
    
    out_buf.seek(0)
    img_str = base64.b64encode(out_buf.getvalue()).decode('utf-8')
    
    # Close the figure to free memory
    plt.close(fig)
    
    return f'data:{mime_type};base64,{img_str}'

# Helper function to sample large datasets
def sample_dataframe(df, max_points=100):
    """Sample dataframe if it's too large."""
    if len(df) > max_points:
        return df.sample(n=max_points, random_state=42)
    return df

# Routes
@router.get("/dashboard", response_class=HTMLResponse)
@cache(expire=60 * 15)  # Cache for 15 minutes
async def analytics_dashboard():
    """Serves a comprehensive analytics dashboard with lazy-loading for better performance."""
    try:
        # Fetch all performance data
        performances = await mongo_service.get_performances_optimized()
        
        if not performances:
            return "<h1>No data available yet</h1>"
        
        # Convert to DataFrame and anonymize
        df = pd.DataFrame(performances)
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
        
        # Calculate key metrics for overview page
        participant_count = len(df)
        avg_score = round(df['performanceScore'].mean(), 2)
        avg_evacuation_time = round(df[event_columns].sum(axis=1).mean(), 2)
        success_rate = round((df['performanceScore'] > 0).mean() * 100, 2)
        
        # Age-performance correlation
        age_perf_corr = 0
        if 'numeric_age' in df.columns:
            age_perf_corr = round(df['numeric_age'].corr(df['performanceScore']), 2)
        
        # Generate HTML dashboard with lazy loading for tabs
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Inferno VR Fire Safety Research Dashboard</title>
            <link rel="icon" href="/static/favicon.ico" type="image/x-icon">
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
                .spinner {
                    border: 4px solid rgba(0, 0, 0, 0.1);
                    width: 36px;
                    height: 36px;
                    border-radius: 50%;
                    border-left-color: #09f;
                    animation: spin 1s linear infinite;
                    margin: 30px auto;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                .loading { text-align: center; padding: 50px; font-size: 18px; color: #666; }
            </style>
            <script>
                // Lazy loading tabs
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
                    
                    // Lazy load tab content when clicked
                    if (tabName !== 'overview' && !document.getElementById(tabName).dataset.loaded) {
                        showSpinner(tabName);
                        fetch(`/api/dashboard-tab/${tabName}`)
                            .then(response => response.text())
                            .then(html => {
                                document.getElementById(tabName).innerHTML = html;
                                document.getElementById(tabName).dataset.loaded = "true";
                                initLazyImages();
                            })
                            .catch(error => {
                                document.getElementById(tabName).innerHTML = `<p>Error loading content: ${error}</p>`;
                            });
                    }
                }
                
                function showSpinner(tabId) {
                    document.getElementById(tabId).innerHTML = '<div class="spinner"></div><p style="text-align:center">Loading visualizations...</p>';
                }
                
                // Lazy load images
                function initLazyImages() {
                    var lazyImages = [].slice.call(document.querySelectorAll("img.lazy"));
                    
                    if ("IntersectionObserver" in window) {
                        let lazyImageObserver = new IntersectionObserver(function(entries, observer) {
                            entries.forEach(function(entry) {
                                if (entry.isIntersecting) {
                                    let lazyImage = entry.target;
                                    lazyImage.src = lazyImage.dataset.src;
                                    lazyImage.classList.remove("lazy");
                                    lazyImageObserver.unobserve(lazyImage);
                                }
                            });
                        });
                        
                        lazyImages.forEach(function(lazyImage) {
                            lazyImageObserver.observe(lazyImage);
                        });
                    } else {
                        // Fallback for browsers without IntersectionObserver
                        lazyImages.forEach(function(lazyImage) {
                            lazyImage.src = lazyImage.dataset.src;
                            lazyImage.classList.remove("lazy");
                        });
                    }
                }
                
                // Initialize when page loads
                document.addEventListener("DOMContentLoaded", function() {
                    initLazyImages();
                });
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
                    
                    <div id="overview" class="tab-content" style="display: block;" data-loaded="true">
        """
        
        # Overview Tab Content (always loaded)
        # Key metrics
        html += """
                        <h2>Key Metrics</h2>
                        <div class="metrics">
        """
        
        # Total participants
        html += f"""
                            <div class="metric-card">
                                <div class="metric-value">{participant_count}</div>
                                <div class="metric-label">Total Participants</div>
                            </div>
        """
        
        # Average performance score
        html += f"""
                            <div class="metric-card">
                                <div class="metric-value">{avg_score}</div>
                                <div class="metric-label">Average Performance Score</div>
                            </div>
        """
        
        # Average evacuation time
        html += f"""
                            <div class="metric-card">
                                <div class="metric-value">{avg_evacuation_time} s</div>
                                <div class="metric-label">Average Evacuation Time</div>
                            </div>
        """
        
        # Success rate
        html += f"""
                            <div class="metric-card">
                                <div class="metric-value">{success_rate}%</div>
                                <div class="metric-label">Success Rate</div>
                            </div>
        """
        
        html += """
                        </div>
        """
        
        # Performance distribution chart - optimized
        fig = plt.figure(figsize=(8, 5), dpi=80)
        sns.histplot(df['performanceScore'], kde=True, bins=15)
        plt.title('Distribution of Performance Scores')
        plt.xlabel('Performance Score')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        img_data = optimize_plot(fig, 'Performance Score Distribution')
        
        html += f"""
                        <div class="chart-container">
                            <h3>Performance Score Distribution</h3>
                            <img src="{img_data}" width="100%" alt="Performance distribution">
                        </div>
        """
        
        # Time metrics comparison - optimized
        fig = plt.figure(figsize=(8, 5), dpi=80)
        time_means = df[event_columns].mean().sort_values()
        sns.barplot(x=time_means.values, y=time_means.index)
        plt.title('Average Time by Task')
        plt.xlabel('Time (seconds)')
        
        img_data = optimize_plot(fig, 'Average Time by Task')
        
        html += f"""
                        <div class="chart-container">
                            <h3>Average Time by Task</h3>
                            <img src="{img_data}" width="100%" alt="Time by task">
                        </div>
        """
        
        # Latest records (no graph, just table)
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
            # Just show 5 latest records for performance
            latest_records = df.sort_values('timestamp', ascending=False).head(5)
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
        
        # Empty placeholders for other tabs - will be loaded on demand
        html += """
                    <div id="demographics" class="tab-content">
                        <!-- Will be loaded on demand -->
                    </div>
                    <div id="performance" class="tab-content">
                        <!-- Will be loaded on demand -->
                    </div>
                    <div id="correlations" class="tab-content">
                        <!-- Will be loaded on demand -->
                    </div>
                    <div id="time-analysis" class="tab-content">
                        <!-- Will be loaded on demand -->
                    </div>
                </div>
                
                <div style="margin-top: 30px;">
                    <h2>Research Insights</h2>
                    <p>This dashboard provides anonymized data from participants in the Inferno VR Fire-Safety Training program. The data presented here forms the basis for research on Immersive Navigation for Fire Emergency Response & Neutralization Operations.</p>
                    <p>Key insights:</p>
                    <ul>
                        <li>The average performance score across all participants is {avg_score:.2f}</li>
        """
        
        if 'numeric_age' in df.columns:
            html += f"""
                        <li>The relationship between age and performance shows {age_perf_corr:.2f} correlation</li>
            """
            
        html += f"""
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
        import traceback
        error_details = traceback.format_exc()
        return f"""
        <h1>Error generating dashboard</h1>
        <p>{str(e)}</p>
        <pre>{error_details}</pre>
        """

@router.get("/dashboard-tab/{tab_name}", response_class=HTMLResponse)
@cache(expire=60 * 15)  # Cache tab content too
async def get_dashboard_tab_content(tab_name: str):
    """Serve individual tab content for lazy loading."""
    try:
        # Fetch data
        performances = await mongo_service.get_performances_optimized()
        if not performances:
            return "<p>No data available</p>"
        
        # Prepare data (same as in main dashboard)
        df = pd.DataFrame(performances)
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
        
        # Generate content for the requested tab
        if tab_name == "demographics":
            return generate_demographics_tab(df, event_columns)
        elif tab_name == "performance":
            return generate_performance_tab(df, event_columns)
        elif tab_name == "correlations":
            return generate_correlations_tab(df, event_columns)
        elif tab_name == "time-analysis":
            return generate_time_analysis_tab(df, event_columns)
        else:
            return "<p>Unknown tab</p>"
            
    except Exception as e:
        return f"<p>Error loading tab: {str(e)}</p>"

def generate_demographics_tab(df, event_columns):
    """Generate HTML content for demographics tab."""
    html = "<h2>Demographic Analysis</h2>"
    
    # Age distribution
    if 'numeric_age' in df.columns:
        fig = plt.figure(figsize=(8, 5), dpi=80)
        sns.histplot(df['numeric_age'], kde=True, bins=15)
        plt.title('Age Distribution of Participants')
        plt.xlabel('Age')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        img_data = optimize_plot(fig, 'Age Distribution')
        
        html += f"""
            <div class="chart-container">
                <h3>Age Distribution</h3>
                <img class="lazy" data-src="{img_data}" width="100%" alt="Age distribution">
            </div>
        """
    
    # Performance by age group
    if 'age_group' in df.columns:
        fig = plt.figure(figsize=(8, 5), dpi=80)
        sns.boxplot(x='age_group', y='performanceScore', data=df)
        plt.title('Performance Score by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Performance Score')
        plt.xticks(rotation=45)
        
        img_data = optimize_plot(fig, 'Performance by Age Group')
        
        html += f"""
            <div class="chart-container">
                <h3>Performance by Age Group</h3>
                <img class="lazy" data-src="{img_data}" width="100%" alt="Performance by age group">
            </div>
        """
    
    # Scene type distribution
    if 'sceneType' in df.columns:
        scene_counts = df['sceneType'].value_counts()
        fig = plt.figure(figsize=(8, 5), dpi=80)
        plt.pie(scene_counts, labels=scene_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Distribution by Scene Type')
        plt.axis('equal')
        
        img_data = optimize_plot(fig, 'Scene Type Distribution')
        
        html += f"""
            <div class="chart-container">
                <h3>Scene Type Distribution</h3>
                <img class="lazy" data-src="{img_data}" width="100%" alt="Scene type distribution">
            </div>
        """
    
    return html

def generate_performance_tab(df, event_columns):
    """Generate HTML content for performance tab."""
    html = "<h2>Performance Analysis</h2>"
    
    # Performance by scene type
    if 'sceneType' in df.columns:
        fig = plt.figure(figsize=(8, 5), dpi=80)
        sns.boxplot(x='sceneType', y='performanceScore', data=df)
        plt.title('Performance Score by Scene Type')
        plt.xlabel('Scene Type')
        plt.ylabel('Performance Score')
        plt.xticks(rotation=45)
        
        img_data = optimize_plot(fig, 'Performance by Scene Type')
        
        html += f"""
            <div class="chart-container">
                <h3>Performance by Scene Type</h3>
                <img class="lazy" data-src="{img_data}" width="100%" alt="Performance by scene type">
            </div>
        """
    
    # Performance by difficulty
    if 'difficulty' in df.columns:
        fig = plt.figure(figsize=(8, 5), dpi=80)
        sns.boxplot(x='difficulty', y='performanceScore', data=df)
        plt.title('Performance Score by Difficulty Level')
        plt.xlabel('Difficulty')
        plt.ylabel('Performance Score')
        
        img_data = optimize_plot(fig, 'Performance by Difficulty')
        
        html += f"""
            <div class="chart-container">
                <h3>Performance by Difficulty Level</h3>
                <img class="lazy" data-src="{img_data}" width="100%" alt="Performance by difficulty">
            </div>
        """
    
    # Performance comparison by tasks - use sampled data if large
    fig = plt.figure(figsize=(8, 5), dpi=80)
    
    # Sample the data if there are too many points
    sample_df = sample_dataframe(df, max_points=200)
    
    task_performance = pd.DataFrame()
    for col in event_columns:
        task_performance[col] = sample_df[col] / sample_df[col].max()  # Normalize for comparison
    
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
    
    img_data = optimize_plot(fig, 'Task Performance Comparison')
    
    html += f"""
        <div class="chart-container">
            <h3>Task Performance Comparison</h3>
            <img class="lazy" data-src="{img_data}" width="100%" alt="Task performance">
        </div>
    """
    
    return html

def generate_correlations_tab(df, event_columns):
    """Generate HTML content for correlations tab."""
    html = "<h2>Correlation Analysis</h2>"
    
    # Correlation heatmap
    correlation_cols = ['numeric_age'] + event_columns + ['performanceScore'] if 'numeric_age' in df.columns else event_columns + ['performanceScore']
    correlation_df = df[correlation_cols].copy()
    
    # Calculate correlation matrix
    corr_matrix = correlation_df.corr()
    
    # Limit the size of the heatmap for better performance
    fig = plt.figure(figsize=(10, 8), dpi=80)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Matrix')
    
    img_data = optimize_plot(fig, 'Correlation Matrix')
    
    html += f"""
        <div class="chart-container">
            <h3>Correlation Heatmap</h3>
            <img class="lazy" data-src="{img_data}" width="100%" alt="Correlation heatmap">
        </div>
    """
    
    # Scatter plots for important correlations
    if 'numeric_age' in df.columns:
        # Sample data for scatterplots if large
        sample_df = sample_dataframe(df, max_points=150)
        
        # Age vs Performance
        fig = plt.figure(figsize=(8, 5), dpi=80)
        sns.scatterplot(x='numeric_age', y='performanceScore', data=sample_df)
        plt.title('Age vs Performance Score')
        plt.xlabel('Age')
        plt.ylabel('Performance Score')
        plt.grid(True, alpha=0.3)
        
        img_data = optimize_plot(fig, 'Age vs Performance')
        
        html += f"""
            <div class="chart-container">
                <h3>Age vs Performance</h3>
                <img class="lazy" data-src="{img_data}" width="100%" alt="Age vs performance">
            </div>
        """
    
    # Instead of pairplot, use a simpler grid of scatterplots
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=80)
    axes = axes.flatten()
    
    # Sample data for these plots
    sample_df = sample_dataframe(df, max_points=100)
    
    # Plot the most important correlations
    for i, (x, y) in enumerate(itertools.combinations(event_columns, 2)):
        if i >= len(axes):
            break
        sns.scatterplot(x=x, y=y, data=sample_df, ax=axes[i], s=30, alpha=0.6)
        axes[i].set_title(f"{x} vs {y}")
    
    # Hide any remaining subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    img_data = optimize_plot(fig, 'Relationships Between Time Metrics')
    
    html += f"""
        <div class="chart-container">
            <h3>Relationships Between Time Metrics</h3>
            <img class="lazy" data-src="{img_data}" width="100%" alt="Time metric relationships">
        </div>
    """
    
    return html

def generate_time_analysis_tab(df, event_columns):
    """Generate HTML content for time analysis tab."""
    html = "<h2>Time Series Analysis</h2>"
    
    if 'timestamp' in df.columns:
        # Performance over time
        df_sorted = df.sort_values('timestamp')
        
        # Sample if too many data points
        if len(df_sorted) > 150:
            # Keep the newest and oldest points, and sample in between
            newest = df_sorted.tail(50)
            oldest = df_sorted.head(50)
            middle = df_sorted.iloc[50:-50].sample(n=min(50, len(df_sorted)-100))
            df_sorted = pd.concat([oldest, middle, newest]).sort_values('timestamp')
        
        fig = plt.figure(figsize=(10, 5), dpi=80)
        plt.plot(df_sorted['timestamp'], df_sorted['performanceScore'], marker='o', alpha=0.7, markersize=4)
        plt.title('Performance Scores Over Time')
        plt.xlabel('Time')
        plt.ylabel('Performance Score')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        img_data = optimize_plot(fig, 'Performance Trend')
        
        html += f"""
            <div class="chart-container">
                <h3>Performance Trend</h3>
                <img class="lazy" data-src="{img_data}" width="100%" alt="Performance trend">
            </div>
        """
        
        # Average performance by day
        if 'day' in df.columns:
            daily_avg = df.groupby('day')['performanceScore'].mean()
            
            # Limit number of days if too many
            if len(daily_avg) > 20:
                # Sample days but keep the newest ones
                day_indices = list(daily_avg.index)
                selected_days = day_indices[-10:]  # Keep newest 10 days
                if len(day_indices) > 10:
                    # Sample from earlier days
                    sampled_days = sorted(np.random.choice(day_indices[:-10], size=min(10, len(day_indices)-10), replace=False))
                    selected_days = sorted(sampled_days + selected_days)
                daily_avg = daily_avg.loc[selected_days]
            
            fig = plt.figure(figsize=(10, 5), dpi=80)
            plt.plot(daily_avg.index, daily_avg.values, marker='o', linestyle='-')
            plt.title('Average Daily Performance')
            plt.xlabel('Date')
            plt.ylabel('Average Performance Score')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            img_data = optimize_plot(fig, 'Average Daily Performance')
            
            html += f"""
                <div class="chart-container">
                    <h3>Average Daily Performance</h3>
                    <img class="lazy" data-src="{img_data}" width="100%" alt="Daily performance">
                </div>
            """
        
        # Simplified task time improvement chart
        # Instead of 4 plots, show one combined plot
        fig = plt.figure(figsize=(10, 6), dpi=80)
        
        # Sample data points for cleaner visualization
        sample_df = sample_dataframe(df_sorted, max_points=100)
        
        for col in event_columns:
            # Create a rolling average (window size adjusts to data size)
            window_size = max(3, min(5, len(sample_df) // 10))
            sample_df[f'{col}_rolling'] = sample_df[col].rolling(window=window_size, min_periods=1).mean()
            plt.plot(sample_df['timestamp'], sample_df[f'{col}_rolling'], '-', label=col)
        
        plt.title('Task Times Trend')
        plt.xlabel('Time')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        img_data = optimize_plot(fig, 'Task Times Trend')
        
        html += f"""
            <div class="chart-container">
                <h3>Task Times Trend</h3>
                <img class="lazy" data-src="{img_data}" width="100%" alt="Task times trend">
            </div>
        """
    
    return html
