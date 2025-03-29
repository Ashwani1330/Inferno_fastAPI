import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from datetime import datetime, timedelta
from PIL import Image
import logging
import json
import os
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache
from scipy import stats

from services.mongo_service import MongoService
from utils.helpers import parse_age, clean_for_json

logger = logging.getLogger(__name__)

class AnalyticsService:
    def __init__(self):
        self.mongo_service = MongoService()
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "analytics")
        self.plots_dir = os.path.join(self.cache_dir, "plots")
        self.stats_path = os.path.join(self.cache_dir, "dashboard_stats.json")
        self.html_cache_path = os.path.join(self.cache_dir, "dashboard.html")
        
        # Ensure cache directories exist
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Last processed record count (for incremental updates)
        self.last_processed_count = 0
        self.last_update_time = None

    async def should_regenerate(self):
        """Determine if analytics need to be regenerated"""
        try:
            # If cache doesn't exist yet, regenerate
            if not os.path.exists(self.stats_path):
                return True
                
            # Check if new data is available
            current_count = await self.mongo_service.get_performance_count()
            
            # If significant new data or it's been a while, regenerate
            if current_count > self.last_processed_count + 5:
                return True
                
            # Check time-based regeneration (e.g., daily)
            if self.last_update_time:
                time_diff = datetime.now() - self.last_update_time
                if time_diff.total_seconds() > 86400:  # 24 hours
                    return True
                    
            return False
        except Exception as e:
            logger.error(f"Error checking regeneration status: {str(e)}")
            return True

    def optimize_plot(self, fig, title, dpi=80, quality=80, format='webp', max_width=800):
        """Optimize a matplotlib figure for web display and save to file system"""
        plt.title(title)
        plt.tight_layout()
        
        # Save figure to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, 
                    bbox_inches='tight', pad_inches=0.1, transparent=False)
        buf.seek(0)
        
        # Further optimize with PIL
        img = Image.open(buf)
        
        # Resize if larger than max_width
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.LANCZOS)
        
        # Generate filename from title
        filename = title.lower().replace(' ', '_').replace('-', '_')
        filepath = os.path.join(self.plots_dir, f"{filename}.{format}")
        
        # Save as optimized format
        if format == 'webp':
            img.save(filepath, format='WEBP', quality=quality)
            mime_type = 'image/webp'
        else:
            img.save(filepath, format='PNG', optimize=True, quality=quality)
            mime_type = 'image/png'
        
        # Also create a base64 version for direct embedding
        out_buf = io.BytesIO()
        if format == 'webp':
            img.save(out_buf, format='WEBP', quality=quality)
        else:
            img.save(out_buf, format='PNG', optimize=True, quality=quality)
        
        out_buf.seek(0)
        img_str = base64.b64encode(out_buf.getvalue()).decode('utf-8')
        
        # Close the figure to free memory
        plt.close(fig)
        
        return {
            'filepath': filepath,
            'base64': f'data:{mime_type};base64,{img_str}',
            'filename': f"{filename}.{format}"
        }

    async def generate_alternative_visualizations(self, df=None):
        """Generate alternative visualizations to replace the scatterplot matrix"""
        try:
            from services.analysis_service import AnalysisService
            analysis_service = AnalysisService()
            
            if df is None:
                performances = await self.mongo_service.get_performances_optimized()
                if not performances:
                    logger.warning("No performance data available for alternative visualizations")
                    return None
                df = pd.DataFrame(performances)
            
            logger.info(f"Generating alternative visualizations with {len(df)} records")
            
            # Initialize results dictionary
            alternative_visualizations = {}
            
            # Clean data - replace negatives with NaN
            event_columns = ['timeToFindExtinguisher', 'timeToExtinguishFire', 
                           'timeToTriggerAlarm', 'timeToFindExit']
            for col in event_columns:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)
            
            # 1. Generate enhanced correlation heatmap
            try:
                correlation_heatmap = analysis_service.generate_enhanced_correlation_heatmap(df)
                if correlation_heatmap:
                    alternative_visualizations['correlation_heatmap'] = correlation_heatmap
                    logger.info("Successfully generated enhanced correlation heatmap")
                else:
                    logger.warning("Could not generate enhanced correlation heatmap")
            except Exception as e:
                logger.error(f"Error generating enhanced correlation heatmap: {str(e)}")
            
            # New: Generate scatterplot matrix
            try:
                scatter_matrix = analysis_service.generate_scatterplot_matrix(df)
                if scatter_matrix:
                    alternative_visualizations['scatterplot_matrix'] = scatter_matrix
                    logger.info("Successfully generated scatterplot matrix")
                else:
                    logger.warning("Could not generate scatterplot matrix")
            except Exception as e:
                logger.error(f"Error generating scatterplot matrix: {str(e)}")
            
            # --- New alternative visualizations ---
            try:
                perf_trend = analysis_service.generate_performance_trend(df)
                if perf_trend:
                    alternative_visualizations['performance_trend_over_time'] = perf_trend
                    logger.info("Successfully generated performance trend over time")
            except Exception as e:
                logger.error(f"Error generating performance trend: {str(e)}")
            
            try:
                task_speed = analysis_service.generate_task_completion_speed(df)
                if task_speed:
                    alternative_visualizations['performance_by_task_completion_speed'] = task_speed
                    logger.info("Successfully generated task completion speed visualization")
            except Exception as e:
                logger.error(f"Error generating task completion speed visualization: {str(e)}")
            
            try:
                feat_impact = analysis_service.generate_feature_impact(df)
                if feat_impact:
                    alternative_visualizations['feature_impact'] = feat_impact
                    logger.info("Successfully generated feature impact visualization")
            except Exception as e:
                logger.error(f"Error generating feature impact visualization: {str(e)}")
            
            return alternative_visualizations
            
        except Exception as e:
            logger.error(f"Error generating alternative visualizations: {str(e)}", exc_info=True)
            return None

    # Add this stub method for backward compatibility
    async def generate_scatterplot_matrix(self, df=None):
        """Stub method for backward compatibility - redirects to alternative visualizations"""
        logger.info("generate_scatterplot_matrix called - using alternative visualizations instead")
        return await self.generate_alternative_visualizations(df)

    async def process_data(self):
        """Process the performance data and generate analytics"""
        try:
            # Fetch performances
            performances = await self.mongo_service.get_performances_optimized()
            
            if not performances:
                logger.warning("No performance data available")
                return False
                
            # Update processing metadata
            self.last_processed_count = len(performances)
            self.last_update_time = datetime.now()
            
            # Convert to DataFrame and process
            df = pd.DataFrame(performances)
            
            # Anonymize data
            if 'email' in df.columns:
                df['user_id'] = [f"User_{i+1}" for i in range(len(df))]
                df = df.drop('email', axis=1)
            
            # Clean data
            event_columns = ['timeToFindExtinguisher', 'timeToExtinguishFire', 
                            'timeToTriggerAlarm', 'timeToFindExit']
            for col in event_columns:
                df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)
            
            # Process timestamps 
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date'] = df['timestamp'].dt.date
                df['week'] = df['timestamp'].dt.isocalendar().week
                df['month'] = df['timestamp'].dt.month
            
            # Add age groups for deeper analysis
            if 'age' in df.columns:
                df['numeric_age'] = df['age'].apply(lambda x: parse_age(x) if isinstance(x, str) else x)
                df['age_group'] = pd.cut(df['numeric_age'], 
                                         bins=[0, 18, 30, 45, 60, 100], 
                                         labels=['Under 18', '18-30', '31-45', '46-60', 'Over 60'], 
                                         include_lowest=True)
            
            # Add total time column for analysis
            df['total_time'] = df[event_columns].sum(axis=1)
            
            # Calculate key metrics
            stats = {}
            stats['participant_count'] = len(df)
            stats['avg_score'] = round(df['performanceScore'].mean(), 2)
            stats['avg_evacuation_time'] = round(df[event_columns].sum(axis=1).mean(), 2)
            stats['success_rate'] = round((df['performanceScore'] > 0).mean() * 100, 2)
            stats['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Add correlation data
            if 'numeric_age' in df.columns:
                stats['age_performance_correlation'] = round(df['numeric_age'].corr(df['performanceScore']), 2)
            
            # Generate all plots and save their paths
            plots = {}
            
            # ----- OVERVIEW TAB -----
            
            # Performance distribution chart
            fig = plt.figure(figsize=(8, 5), dpi=80)
            sns.histplot(df['performanceScore'], kde=True, bins=15, color='steelblue')
            plt.xlabel('Performance Score')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            plots['performance_distribution'] = self.optimize_plot(fig, 'Distribution of Performance Scores')
            
            # Time metrics comparison
            fig = plt.figure(figsize=(10, 6))
            time_means = df[event_columns].mean().sort_values()
            ax = sns.barplot(x=time_means.values, y=time_means.index, palette='viridis')
            plt.xlabel('Time (seconds)')
            # Add value labels to bars
            for i, val in enumerate(time_means.values):
                ax.text(val + 0.5, i, f'{val:.1f}s', va='center')
            plots['avg_time_by_task'] = self.optimize_plot(fig, 'Average Time by Task')
            
            # Recent participation trend
            if 'date' in df.columns:
                # Get last 30 days for trend
                last_30_days = datetime.now().date() - timedelta(days=30)
                trend_df = df[df['date'] >= last_30_days].copy()
                daily_counts = trend_df.groupby('date').size()
                
                # Fill missing dates with zeros
                date_range = pd.date_range(start=last_30_days, end=datetime.now().date())
                daily_counts = daily_counts.reindex(date_range, fill_value=0)
                
                fig = plt.figure(figsize=(10, 6))
                plt.plot(daily_counts.index, daily_counts.values, marker='o', linestyle='-', color='royalblue')
                plt.xlabel('Date')
                plt.ylabel('Number of Participants')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plots['participation_trend'] = self.optimize_plot(fig, 'Participation Trend')
            
            # ----- DEMOGRAPHICS TAB -----
            
            # Age distribution
            if 'numeric_age' in df.columns:
                fig = plt.figure(figsize=(10, 6))
                sns.histplot(df['numeric_age'], kde=True, bins=20, color='forestgreen')
                plt.xlabel('Age')
                plt.ylabel('Count')
                plt.grid(True, alpha=0.3)
                
                # Add mean and median lines
                mean_age = df['numeric_age'].mean()
                median_age = df['numeric_age'].median()
                plt.axvline(mean_age, color='red', linestyle='--', label=f'Mean: {mean_age:.1f}')
                plt.axvline(median_age, color='blue', linestyle=':', label=f'Median: {median_age:.1f}')
                plt.legend()
                
                plots['age_distribution'] = self.optimize_plot(fig, 'Age Distribution of Participants')
            
            # Performance by age group
            if 'age_group' in df.columns:
                fig = plt.figure(figsize=(10, 6))
                ax = sns.boxplot(x='age_group', y='performanceScore', data=df, palette='viridis')
                plt.xlabel('Age Group')
                plt.ylabel('Performance Score')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                # Add average line
                avg_score = df['performanceScore'].mean()
                plt.axhline(y=avg_score, color='red', linestyle='--', 
                           label=f'Overall Average: {avg_score:.2f}')
                plt.legend()
                
                # Add count of participants in each group
                group_counts = df['age_group'].value_counts().sort_index()
                for i, count in enumerate(group_counts):
                    ax.text(i, df['performanceScore'].min(), f'n={count}', 
                            ha='center', va='bottom', color='black')
                
                plots['performance_by_age'] = self.optimize_plot(fig, 'Performance Score by Age Group')
                
                # Time metrics by age group
                fig, axs = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle('Time Metrics by Age Group', fontsize=16)
                
                for i, col in enumerate(event_columns):
                    row, col_idx = divmod(i, 2)
                    sns.boxplot(x='age_group', y=col, data=df, ax=axs[row, col_idx], palette='viridis')
                    axs[row, col_idx].set_title(col)
                    axs[row, col_idx].set_xlabel('Age Group')
                    axs[row, col_idx].set_ylabel('Time (seconds)')
                    axs[row, col_idx].tick_params(axis='x', rotation=45)
                    # Add mean line for comparison
                    mean_val = df[col].mean()
                    axs[row, col_idx].axhline(y=mean_val, color='red', linestyle='--')
                
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plots['time_by_age'] = self.optimize_plot(fig, 'Time Metrics by Age Group')
            
            # ----- PERFORMANCE ANALYSIS TAB -----
            
            # Performance by difficulty
            if 'difficulty' in df.columns:
                fig = plt.figure(figsize=(10, 6))
                ax = sns.barplot(x='difficulty', y='performanceScore', data=df, palette='YlOrRd', 
                                errorbar='sd', errwidth=1.5)
                plt.xlabel('Difficulty Level')
                plt.ylabel('Average Performance Score')
                plt.title('Performance by Difficulty Level')
                plt.grid(True, alpha=0.3)
                
                # Add data labels
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.1f}', 
                               (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='bottom')
                
                # Add count of participants
                difficulty_counts = df['difficulty'].value_counts()
                for i, diff in enumerate(difficulty_counts.index):
                    count = difficulty_counts[diff]
                    if i < len(ax.patches):
                        ax.text(i, 0, f'n={count}', ha='center', va='bottom')
                
                plots['performance_by_difficulty'] = self.optimize_plot(fig, 'Performance by Difficulty Level')
            
            # Performance by scene type
            if 'sceneType' in df.columns:
                fig = plt.figure(figsize=(10, 6))
                ax = sns.barplot(x='sceneType', y='performanceScore', data=df, palette='Blues_d',
                                errorbar='sd', errwidth=1.5)
                plt.xlabel('Scene Type')
                plt.ylabel('Average Performance Score')
                plt.title('Performance by Scene Type')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                # Add data labels
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.1f}', 
                               (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='bottom')
                
                # Add count of participants
                scene_counts = df['sceneType'].value_counts()
                for i, scene in enumerate(scene_counts.index):
                    count = scene_counts[scene]
                    if i < len(ax.patches):
                        ax.text(i, 0, f'n={count}', ha='center', va='bottom')
                
                plots['performance_by_scene'] = self.optimize_plot(fig, 'Performance by Scene Type')
            
            # Success rate comparison
            success_rates = {}
            if 'difficulty' in df.columns:
                success_rates['difficulty'] = df.groupby('difficulty')['performanceScore'].apply(
                    lambda x: (x > 0).mean() * 100).to_dict()
            if 'sceneType' in df.columns:
                success_rates['scene'] = df.groupby('sceneType')['performanceScore'].apply(
                    lambda x: (x > 0).mean() * 100).to_dict()
            if 'age_group' in df.columns:
                success_rates['age'] = df.groupby('age_group')['performanceScore'].apply(
                    lambda x: (x > 0).mean() * 100).to_dict()
                
            stats['success_rates'] = success_rates
            
            # Generate alternative visualizations
            if len(df) > 5:
                alternative_viz = await self.generate_alternative_visualizations(df)
                if alternative_viz:
                    # Add each alternative visualization to plots
                    for viz_key, viz_data in alternative_viz.items():
                        plots[viz_key] = viz_data
                    logger.info(f"Added {len(alternative_viz)} alternative visualizations")
            
            # Weekly/Monthly trends
            time_metrics = {}
            if 'week' in df.columns and len(df['week'].unique()) > 1:
                weekly_perf = df.groupby('week')['performanceScore'].agg(['mean', 'count']).reset_index()
                weekly_perf['period'] = weekly_perf['week'].apply(lambda x: f'Week {x}')
                time_metrics['weekly'] = weekly_perf[['period', 'mean', 'count']].to_dict('records')
                
                # Weekly performance plot
                fig = plt.figure(figsize=(12, 6))
                ax = sns.barplot(x='week', y='mean', data=weekly_perf, color='mediumseagreen')
                plt.xlabel('Week Number')
                plt.ylabel('Average Performance Score')
                plt.title('Weekly Performance Averages')
                plt.grid(True, alpha=0.3)
                
                # Add count labels
                for i, row in weekly_perf.iterrows():
                    ax.text(i, row['mean'] + 1, f"n={row['count']}", ha='center')
                
                plots['weekly_performance'] = self.optimize_plot(fig, 'Weekly Performance Averages')
            
            if 'month' in df.columns and len(df['month'].unique()) > 1:
                monthly_perf = df.groupby('month')['performanceScore'].agg(['mean', 'count']).reset_index()
                monthly_perf['period'] = monthly_perf['month'].apply(lambda x: f'Month {x}')
                time_metrics['monthly'] = monthly_perf[['period', 'mean', 'count']].to_dict('records')
                
                # Monthly performance plot
                fig = plt.figure(figsize=(12, 6))
                ax = sns.barplot(x='month', y='mean', data=monthly_perf, color='coral')
                plt.xlabel('Month')
                plt.ylabel('Average Performance Score')
                plt.title('Monthly Performance Averages')
                plt.grid(True, alpha=0.3)
                
                # Add count labels
                for i, row in monthly_perf.iterrows():
                    ax.text(i, row['mean'] + 1, f"n={row['count']}", ha='center')
                
                plots['monthly_performance'] = self.optimize_plot(fig, 'Monthly Performance Averages')
            
            stats['time_metrics'] = time_metrics
            
            # Task time trends
            if 'date' in df.columns and len(df['date'].unique()) > 5:
                fig, axs = plt.subplots(len(event_columns), 1, figsize=(12, 4*len(event_columns)))
                
                for i, col in enumerate(event_columns):
                    daily_time = df.groupby('date')[col].mean()
                    
                    if len(event_columns) > 1:
                        ax = axs[i]
                    else:
                        ax = axs
                        
                    ax.plot(daily_time.index, daily_time.values, marker='o', 
                           linestyle='-', label=col)
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Average Time (s)')
                    ax.set_title(f'Daily Average {col}')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3)
                    
                    # Add trend line
                    if len(daily_time) > 1:
                        z = np.polyfit(range(len(daily_time)), daily_time.values, 1)
                        p = np.poly1d(z)
                        ax.plot(daily_time.index, p(range(len(daily_time))), 
                               linestyle='--', color='red', 
                               label=f'Trend: {"+" if z[0]>0 else ""}{z[0]:.3f} per day')
                        ax.legend()
                
                plt.tight_layout()
                plots['task_time_trends'] = self.optimize_plot(fig, 'Task Time Trends')
            
            # Save latest records for display
            latest_records = []
            if 'timestamp' in df.columns:
                latest_df = df.sort_values('timestamp', ascending=False).head(10)
                for _, row in latest_df.iterrows():
                    record = {
                        'user_id': row.get('user_id', 'Anonymous'),
                        'age': row.get('age', 'N/A'),
                        'scene_type': row.get('sceneType', 'N/A'),
                        'difficulty': row.get('difficulty', 'N/A'),
                        'score': float(row.get('performanceScore', 0)),
                        'date': row['timestamp'].strftime('%Y-%m-%d') if 'timestamp' in row else 'N/A'
                    }
                    latest_records.append(record)
            
            # Store all data in JSON format
            dashboard_data = {
                'stats': stats,
                'plots': plots,
                'latest_records': latest_records
            }
            
            with open(self.stats_path, 'w') as f:
                json.dump(dashboard_data, f)
            
            logger.info(f"Successfully processed analytics data for {len(df)} records")
            return True
            
        except Exception as e:
            logger.error(f"Error processing analytics data: {str(e)}")
            return False
    
    async def get_dashboard_data(self):
        """Retrieve the pre-processed dashboard data"""
        try:
            # Check if we need to regenerate
            if await self.should_regenerate():
                logger.info("Regenerating dashboard data...")
                await self.process_data()
            
            # Load cached data
            if os.path.exists(self.stats_path):
                with open(self.stats_path, 'r') as f:
                    return json.load(f)
            else:
                return None
        except Exception as e:
            logger.error(f"Error retrieving dashboard data: {str(e)}")
            return None

    async def update_dashboard_data(self, dashboard_data):
        """Update the stored dashboard data"""
        try:
            with open(self.stats_path, 'w') as f:
                json.dump(dashboard_data, f)
            return True
        except Exception as e:
            logger.error(f"Error updating dashboard data: {str(e)}")
            return False