import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import matplotlib
matplotlib.use('Agg')  # Ensure non-interactive backend is used
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import logging

logger = logging.getLogger(__name__)

class AnalysisService:
    def generate_metric_graph(self, metric, latest_value, series):
        try:
            # Handle invalid values
            valid_series = series.replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid_series) < 2:
                return f'<div class="alert alert-warning">Insufficient data for {metric} visualization</div>'
                
            plt.figure(figsize=(4, 3))
            plt.hist(valid_series, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
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
        except Exception as e:
            logger.error(f"Error generating metric graph for {metric}: {str(e)}")
            return f'<div class="alert alert-danger">Error generating visualization for {metric}: {str(e)}</div>'

    def compute_percentile(self, series, value):
        try:
            valid_values = series.replace([np.inf, -np.inf], np.nan).dropna().tolist()
            if len(valid_values) < 2:
                return np.nan
            return percentileofscore(valid_values, value)
        except Exception as e:
            logger.error(f"Error computing percentile: {str(e)}")
            return np.nan

    def generate_enhanced_correlation_heatmap(self, df):
        """Generate an enhanced correlation heatmap with significance indicators"""
        try:
            perf_vars = ['performanceScore', 'timeToFindExtinguisher', 'timeToExtinguishFire', 
                        'timeToTriggerAlarm', 'timeToFindExit']
            
            vars_to_use = [var for var in perf_vars if var in df.columns]
            
            if len(vars_to_use) < 2:
                return None
                
            plot_df = df[vars_to_use].copy()
            for col in plot_df.columns:
                if col != 'performanceScore':
                    plot_df[col] = plot_df[col].apply(lambda x: np.nan if x < 0 else x)
            
            # Replace inf values and drop NaNs entirely
            plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(plot_df) < 5:
                return None
            
            display_names = {
                'performanceScore': 'Performance Score',
                'timeToFindExtinguisher': 'Find Extinguisher',
                'timeToExtinguishFire': 'Extinguish Fire',
                'timeToTriggerAlarm': 'Trigger Alarm',
                'timeToFindExit': 'Find Exit'
            }
            
            plot_df = plot_df.rename(columns=display_names)
            
            corr_matrix = plot_df.corr().round(2)
            
            p_values = pd.DataFrame(np.ones((len(corr_matrix), len(corr_matrix))), 
                                   index=corr_matrix.index, columns=corr_matrix.columns)
            
            from scipy.stats import pearsonr
            for i, row in enumerate(corr_matrix.index):
                for j, col in enumerate(corr_matrix.columns):
                    if i != j:
                        stat, p = pearsonr(plot_df[row].values, plot_df[col].values)
                        p_values.loc[row, col] = p
            
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            plt.figure(figsize=(10, 8))
            
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                      square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
            
            for i, row in enumerate(corr_matrix.index):
                for j, col in enumerate(corr_matrix.columns):
                    if i > j:
                        corr_value = corr_matrix.iloc[i, j]
                        p_value = p_values.iloc[i, j]
                        # Add check to skip invalid values
                        if not (np.isfinite(corr_value) and np.isfinite(p_value)):
                            continue
                        if p_value < 0.001:
                            plt.text(j+0.5, i+0.85, '***', ha='center', va='center', color='white' if abs(corr_value) > 0.4 else 'black')
                        elif p_value < 0.01:
                            plt.text(j+0.5, i+0.85, '**', ha='center', va='center', color='white' if abs(corr_value) > 0.4 else 'black')
                        elif p_value < 0.05:
                            plt.text(j+0.5, i+0.85, '*', ha='center', va='center', color='white' if abs(corr_value) > 0.4 else 'black')
            
            plt.title('Enhanced Correlation Matrix with Significance Levels', fontsize=14)
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return {
                'base64': f'data:image/png;base64,{image_base64}',
                'title': 'Enhanced Correlation Matrix'
            }
        except Exception as e:
            logger.error(f"Error generating correlation heatmap: {str(e)}")
            return None

    def generate_scatterplot_matrix(self, df):
        """Generate an optimized scatterplot matrix using seaborn pairplot."""
        try:
            perf_vars = ['performanceScore', 'timeToFindExtinguisher', 'timeToExtinguishFire', 
                        'timeToTriggerAlarm', 'timeToFindExit']
            vars_to_use = [var for var in perf_vars if var in df.columns]
            if len(vars_to_use) < 2:
                return None
                
            # Make a copy to avoid modifying original data
            plot_df = df[vars_to_use].copy()
            
            # Replace negative values with NaN
            for col in plot_df.columns:
                if col != 'performanceScore':
                    plot_df[col] = plot_df[col].apply(lambda x: np.nan if x < 0 else x)
                    
            # Replace inf values and drop all NaNs
            plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(plot_df) < 5:
                logger.warning(f"Not enough valid data points for scatterplot matrix: {len(plot_df)}")
                return None
                
            if len(plot_df) > 100:
                plot_df = plot_df.sample(n=100, random_state=42)
                
            import seaborn as sns
            from io import BytesIO
            import base64
            g = sns.pairplot(plot_df, diag_kind='kde', corner=True, plot_kws={'alpha': 0.6})
            g.fig.suptitle('Scatterplot Matrix', fontsize=14)
            g.fig.tight_layout()
            buf = BytesIO()
            g.fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(g.fig)
            return {
                'base64': f'data:image/png;base64,{image_base64}',
                'title': 'Scatterplot Matrix'
            }
        except Exception as e:
            logger.error(f"Scatterplot matrix generation error: {str(e)}")
            return None

    def generate_task_completion_speed(self, df):
        """Generate a box plot for task completion speed for a professional look."""
        event_cols = ['timeToFindExtinguisher', 'timeToExtinguishFire', 'timeToTriggerAlarm', 'timeToFindExit']
        available = [col for col in event_cols if col in df.columns]
        if not available:
            return None
        # Melt the dataframe for a unified box plot
        df_melt = df[available].melt(var_name='Task', value_name='Time')
        # Remove NaN values to avoid "posx and posy should be finite values" errors
        df_melt = df_melt.replace([np.inf, -np.inf], np.nan).dropna()
        
        plt.figure(figsize=(8, 6))
        # Fix: use category for x variable, not hue with legend=False
        ax = sns.boxplot(x='Task', y='Time', data=df_melt, palette='Set2')
        plt.xlabel('Task')
        plt.ylabel('Completion Time (s)')
        plt.title('Task Completion Speed')
        plt.grid(True, alpha=0.3)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return {
            'base64': f'data:image/png;base64,{image_base64}',
            'title': 'Task Completion Speed'
        }

    def generate_feature_impact(self, df):
        """Generate a bar chart for feature impact based on correlation with performance."""
        features = ['timeToFindExtinguisher', 'timeToExtinguishFire', 'timeToTriggerAlarm', 'timeToFindExit']
        if 'performanceScore' not in df.columns:
            return None
        available = [f for f in features if f in df.columns]
        if not available:
            return None
        
        # Calculate correlations, handling potential NaN values
        correlations = {}
        for feat in available:
            # Drop rows with NaN values for computing correlation
            valid_data = df[[feat, 'performanceScore']].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid_data) > 1:  # Need at least 2 points for correlation
                correlations[feat] = valid_data[feat].corr(valid_data['performanceScore'])
            else:
                correlations[feat] = 0  # Default for insufficient data
        
        corr_series = pd.Series(correlations).sort_values(ascending=False)
        
        # Check if we have valid correlation values
        if corr_series.isnull().all():
            return None
            
        plt.figure(figsize=(8, 6))
        # Fix: use x and y directly instead of hue with legend=False
        ax = sns.barplot(x=corr_series.index, y=corr_series.values, palette='coolwarm')
        plt.xlabel('Feature')
        plt.ylabel('Correlation with Performance')
        plt.title('Feature Impact')
        plt.grid(True, alpha=0.3)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return {
            'base64': f'data:image/png;base64,{image_base64}',
            'title': 'Feature Impact'
        }

    def generate_performance_trend(self, df):
        """Generate a line plot of daily average performance scores with smoothing even if data is sparse."""
        if 'timestamp' not in df.columns or 'performanceScore' not in df.columns:
            return None
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        trend = df.groupby(df['timestamp'].dt.date)['performanceScore'].mean().reset_index()
        trend = trend.sort_values('timestamp')
        
        # Always apply smoothing: use window size 3 if possible, or fallback to 1
        window_size = 3 if len(trend) >= 3 else 1
        trend['smoothed'] = trend['performanceScore'].rolling(window=window_size, min_periods=1, center=True).mean()
        
        plt.figure(figsize=(8, 6))
        plt.plot(trend['timestamp'], trend['smoothed'], marker='o', linestyle='-', color='royalblue', label='Smoothed Trend')
        plt.scatter(trend['timestamp'], trend['performanceScore'], color='gray', alpha=0.5, label='Daily Average')
        plt.xlabel('Date')
        plt.ylabel('Avg. Performance Score')
        plt.title('Longitudinal Performance Trend')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return {
            'base64': f'data:image/png;base64,{image_base64}',
            'title': 'Longitudinal Performance Trend'
        }

    def generate_report(self, df, latest_record):
        event_columns = ['timeToFindExtinguisher', 'timeToExtinguishFire', 'timeToTriggerAlarm', 'timeToFindExit']
        score_column = 'performanceScore'
        
        for col in event_columns:
            df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)
        
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
        
        graphs_html = {}
        for col in ['timeToFindExtinguisher', 'timeToExtinguishFire', 'timeToTriggerAlarm', 'timeToFindExit',
                    'performanceScore']:
            if col in df.columns:
                graphs_html[col] = self.generate_metric_graph(col, latest_record[col], df[col])
        
        enhanced_heatmap = self.generate_enhanced_correlation_heatmap(df)
        if enhanced_heatmap:
            graphs_html['enhanced_heatmap'] = f'<img src="{enhanced_heatmap["base64"]}" alt="Enhanced Correlation Heatmap" style="max-width:600px;"/>'
        
        scatter_matrix = self.generate_scatterplot_matrix(df)
        if scatter_matrix:
            graphs_html['scatterplot_matrix'] = f'<img src="{scatter_matrix["base64"]}" alt="Scatterplot Matrix" style="max-width:600px;"/>'
        
        task_speed = self.generate_task_completion_speed(df)
        if task_speed:
            graphs_html['performance_by_task_completion_speed'] = f'<img src="{task_speed["base64"]}" alt="Task Completion Speed" style="max-width:600px;"/>'
        
        feature_impact = self.generate_feature_impact(df)
        if feature_impact:
            graphs_html['feature_impact'] = f'<img src="{feature_impact["base64"]}" alt="Feature Impact" style="max-width:600px;"/>'
        
        performance_trend = self.generate_performance_trend(df)
        if performance_trend:
            graphs_html['performance_trend'] = f'<img src="{performance_trend["base64"]}" alt="Performance Trend Over Time" style="max-width:600px;"/>'
        
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
            report_html += f"<h3>{metric.replace('_',' ').title()}</h3>{img_html}"
        
        return report_html
