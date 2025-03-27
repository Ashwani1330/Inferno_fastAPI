import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from datetime import datetime
from PIL import Image
import logging
import json
import os
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache

from services.mongo_service import MongoService
from utils.helpers import parse_age

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
            
            # Add age groups for deeper analysis
            if 'age' in df.columns:
                df['numeric_age'] = df['age'].apply(lambda x: parse_age(x) if isinstance(x, str) else x)
                df['age_group'] = pd.cut(df['numeric_age'], 
                                      bins=[0, 18, 30, 45, 60, 100], 
                                      labels=['Under 18', '18-30', '31-45', '46-60', 'Over 60'])
            
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
            
            # Performance distribution chart
            fig = plt.figure(figsize=(8, 5), dpi=80)
            sns.histplot(df['performanceScore'], kde=True)
            plt.xlabel('Performance Score')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            plots['performance_distribution'] = self.optimize_plot(fig, 'Distribution of Performance Scores')
            
            # Time metrics comparison
            fig = plt.figure(figsize=(10, 6))
            time_means = df[event_columns].mean().sort_values()
            sns.barplot(x=time_means.values, y=time_means.index)
            plt.xlabel('Time (seconds)')
            plots['avg_time_by_task'] = self.optimize_plot(fig, 'Average Time by Task')
            
            # Recent participation trend
            if 'date' in df.columns:
                daily_counts = df.groupby('date').size()
                fig = plt.figure(figsize=(10, 6))
                plt.plot(daily_counts.index, daily_counts.values, marker='o')
                plt.xlabel('Date')
                plt.ylabel('Number of Participants')
                plt.xticks(rotation=45)
                plots['participation_trend'] = self.optimize_plot(fig, 'Participation Trend')
            
            # Generate more plots for each tab...
            # Age distribution
            if 'numeric_age' in df.columns:
                fig = plt.figure(figsize=(10, 6))
                sns.histplot(df['numeric_age'], kde=True, bins=20)
                plt.xlabel('Age')
                plt.ylabel('Count')
                plt.grid(True, alpha=0.3)
                plots['age_distribution'] = self.optimize_plot(fig, 'Age Distribution of Participants')
            
            # Performance by age group
            if 'age_group' in df.columns:
                fig = plt.figure(figsize=(10, 6))
                sns.boxplot(x='age_group', y='performanceScore', data=df)
                plt.xlabel('Age Group')
                plt.ylabel('Performance Score')
                plt.xticks(rotation=45)
                plots['performance_by_age'] = self.optimize_plot(fig, 'Performance Score by Age Group')
                
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
                plots['time_by_age'] = self.optimize_plot(fig, 'Time Metrics by Age Group')
            
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
