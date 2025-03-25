import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class AnalysisService:
    def generate_metric_graph(self, metric, latest_value, series):
        plt.figure(figsize=(4, 3))
        plt.hist(series.dropna(), bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(latest_value, color='red', linestyle='dashed', linewidth=2)
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f'<img src="data:image/png;base64,{image_base64}" alt="{metric} graph" style="max-width:600px;"/>'

    def compute_percentile(self, series, value):
        valid_values = series.dropna().tolist()
        if len(valid_values) == 0:
            return np.nan
        return percentileofscore(valid_values, value)

    def generate_report(self, df, latest_record):
        # Clean data
        event_columns = ['timeToFindExtinguisher', 'timeToExtinguishFire', 'timeToTriggerAlarm', 'timeToFindExit']
        score_column = 'performanceScore'
        
        for col in event_columns:
            df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)
        
        # Compute percentiles
        metrics = {}
        all_metrics = event_columns + [score_column]
        
        for col in all_metrics:
            if col in df.columns:
                latest_value = latest_record[col]
                percentile = self.compute_percentile(df[col], latest_value)
                metrics[col] = {
                    "value": latest_value,
                    "percentile": percentile
                }
        
        # Generate graphs
        graphs_html = {}
        for col in all_metrics:
            if col in df.columns:
                graphs_html[col] = self.generate_metric_graph(col, latest_record[col], df[col])
        
        # Create HTML report
        report_html = "<h1>Latest Performance Report</h1>"
        report_html += f"<p>Date: {pd.to_datetime('today').strftime('%Y-%m-%d')}</p>"
        
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
        
        return report_html
