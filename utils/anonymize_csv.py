import pandas as pd
import os
import argparse
import sys
import logging
from anonymization import anonymize_dataframe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def anonymize_csv_file(input_file, output_file=None):
    """
    Anonymize emails in a CSV file.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the anonymized CSV file (defaults to input_anonymized.csv)
    
    Returns:
        Path to the anonymized file
    """
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return None
        
    if not output_file:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_anonymized{ext}"
    
    try:
        logger.info(f"Reading CSV file: {input_file}")
        df = pd.read_csv(input_file, encoding='utf-8')
        
        # Log original column names
        logger.info(f"Original columns: {df.columns.tolist()}")
        logger.info(f"Processing {len(df)} records")
        
        # Anonymize the dataframe
        df_anon = anonymize_dataframe(df)
        
        # Log anonymized column names
        logger.info(f"Anonymized columns: {df_anon.columns.tolist()}")
        
        # Save anonymized data
        df_anon.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Anonymized CSV saved to: {output_file}")
        
        return output_file
    except Exception as e:
        logger.error(f"Error anonymizing CSV: {str(e)}")
        return None

def batch_process_csv_files(directory, output_dir=None):
    """
    Process all CSV files in a directory.
    
    Args:
        directory: Directory containing CSV files
        output_dir: Directory to save anonymized files (defaults to input directory)
    
    Returns:
        Number of successfully processed files
    """
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        return 0
        
    if not output_dir:
        output_dir = directory
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    csv_files = [f for f in os.listdir(directory) if f.lower().endswith('.csv')]
    logger.info(f"Found {len(csv_files)} CSV files in {directory}")
    
    success_count = 0
    for csv_file in csv_files:
        input_path = os.path.join(directory, csv_file)
        output_path = os.path.join(output_dir, f"{os.path.splitext(csv_file)[0]}_anonymized.csv")
        
        if anonymize_csv_file(input_path, output_path):
            success_count += 1
    
    logger.info(f"Successfully anonymized {success_count} out of {len(csv_files)} files")
    return success_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anonymize emails in CSV files")
    parser.add_argument("input", help="Input CSV file or directory containing CSV files")
    parser.add_argument("--output", "-o", help="Output file or directory (optional)")
    parser.add_argument("--batch", "-b", action="store_true", help="Process all CSV files in input directory")
    
    args = parser.parse_args()
    
    if args.batch:
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} is not a directory")
            sys.exit(1)
        batch_process_csv_files(args.input, args.output)
    else:
        anonymize_csv_file(args.input, args.output)
