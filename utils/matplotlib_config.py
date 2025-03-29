"""
Matplotlib configuration utilities to ensure proper non-interactive backend usage
in asynchronous web applications.
"""
import logging
import matplotlib
import numpy as np

logger = logging.getLogger(__name__)

def configure_matplotlib():
    """Configure matplotlib for non-interactive web server use."""
    try:
        # Set the backend to Agg (non-interactive)
        matplotlib.use('Agg', force=True)
        
        # Verify the backend was set correctly
        current_backend = matplotlib.get_backend()
        logger.info(f"Matplotlib backend: {current_backend}")
        
        if current_backend.lower() != 'agg':
            logger.warning(f"Expected 'agg' backend but got '{current_backend}'")
        
        # Disable interactive mode
        import matplotlib.pyplot as plt
        plt.ioff()
        
        # Configure global settings for better web rendering
        matplotlib.rcParams['figure.figsize'] = (8, 6)
        matplotlib.rcParams['figure.dpi'] = 100
        matplotlib.rcParams['savefig.dpi'] = 100
        matplotlib.rcParams['font.size'] = 10
        matplotlib.rcParams['axes.titlesize'] = 12
        matplotlib.rcParams['axes.labelsize'] = 10
        
        return True
    except Exception as e:
        logger.error(f"Error configuring matplotlib: {str(e)}")
        return False

def clean_dataframe_for_plotting(df, columns=None):
    """
    Clean a dataframe for plotting by removing NaN, inf values and
    optionally filtering specific columns.
    
    Args:
        df: DataFrame to clean
        columns: List of column names to include (if None, use all columns)
        
    Returns:
        Cleaned DataFrame
    """
    if df is None or len(df) == 0:
        return None
        
    # Make a copy to avoid modifying the original
    clean_df = df.copy()
    
    # Replace infinities with NaN
    clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
    
    # Filter columns if specified
    if columns:
        available_cols = [col for col in columns if col in clean_df.columns]
        if not available_cols:
            return None
        clean_df = clean_df[available_cols]
    
    # Drop rows with any NaN values
    clean_df = clean_df.dropna()
    
    if len(clean_df) < 2:
        return None
        
    return clean_df

def safe_boxplot(data, x, y, **kwargs):
    """
    Create a safe boxplot that handles NaN and invalid values appropriately.
    
    Args:
        data: DataFrame
        x: x-axis variable name
        y: y-axis variable name
        **kwargs: Additional arguments for seaborn.boxplot
        
    Returns:
        matplotlib axis object or None on error
    """
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # Remove any rows where either x or y is NaN
        plot_data = data.dropna(subset=[x, y])
        
        # Replace infinities with NaN and drop those rows
        plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y])
        
        if len(plot_data) < 5:
            logger.warning(f"Not enough data for boxplot: only {len(plot_data)} valid rows")
            return None
            
        return sns.boxplot(x=x, y=y, data=plot_data, **kwargs)
    except Exception as e:
        logger.error(f"Error creating boxplot: {str(e)}")
        return None
