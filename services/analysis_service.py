import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
import seaborn as sns
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

    def generate_scatterplot_matrix(self, df):
        """Generate a scatterplot matrix for key performance variables"""
        # Select relevant variables for the scatterplot matrix
        perf_vars = ['performanceScore', 'timeToFindExtinguisher', 'timeToExtinguishFire', 
                     'timeToTriggerAlarm', 'timeToFindExit']
        
        # Clean names for display
        display_names = {
            'performanceScore': 'Score',
            'timeToFindExtinguisher': 'Find Extinguisher',
            'timeToExtinguishFire': 'Extinguish Fire',
            'timeToTriggerAlarm': 'Trigger Alarm',
            'timeToFindExit': 'Find Exit'
        }
        
        # Use only variables that exist in the dataframe
        vars_to_use = [var for var in perf_vars if var in df.columns]
        
        if len(vars_to_use) < 2:
            return None
            
        # Clean data - replace negatives with NaN
        plot_df = df[vars_to_use].copy()
        for col in plot_df.columns:
            if col != 'performanceScore':  # Don't filter performance scores
                plot_df[col] = plot_df[col].apply(lambda x: np.nan if x < 0 else x)
        
        # Drop rows with NaN values
        plot_df = plot_df.dropna()
        
        if len(plot_df) < 10:  # Not enough data for meaningful visualization
            return None
            
        # Create scatterplot matrix
        plt.figure(figsize=(12, 10))
        
        # Rename columns for display
        renamed_df = plot_df.rename(columns=display_names)
        
        # Create the scatterplot matrix with Seaborn
        g = sns.pairplot(renamed_df, corner=True, diag_kind='kde', 
                          plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k', 'linewidth': 0.5},
                          diag_kws={'fill': True, 'alpha': 0.6})
        
        # Calculate and add correlation coefficients
        for i, var1 in enumerate(renamed_df.columns):
            for j, var2 in enumerate(renamed_df.columns):
                if j > i:  # Upper triangle only
                    corr = renamed_df[var1].corr(renamed_df[var2]).round(2)
                    g.axes[i, j].annotate(f'r = {corr}', 
                                          xy=(0.5, 0.9), 
                                          xycoords='axes fraction',
                                          ha='center',
                                          va='center',
                                          fontsize=10,
                                          bbox=dict(boxstyle='round,pad=0.5', 
                                                    fc='white', 
                                                    alpha=0.8))
        
        plt.suptitle('Relationships Between Performance Variables', 
                     fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Convert plot to base64 image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return {
            'base64': f'data:image/png;base64,{image_base64}',
            'title': 'Scatterplot Matrix of Performance Variables'
        }

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
        
        # Add scatterplot matrix to the report
        scatterplot_matrix = self.generate_scatterplot_matrix(df)
        if scatterplot_matrix:
            graphs_html['scatterplot_matrix'] = f'<img src="{scatterplot_matrix["base64"]}" alt="Scatterplot Matrix" style="max-width:600px;"/>'
        
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
