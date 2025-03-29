import re
import hashlib
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def anonymize_email(text):
    """
    Anonymize email addresses found in text.
    Preserves domain but replaces username with a hash.
    """
    if not isinstance(text, str):
        return text
        
    # Regex to find email addresses
    email_pattern = r'([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    
    # Find all email addresses in the text
    matches = re.findall(email_pattern, text)
    
    # If no emails found, return original text
    if not matches:
        return text
        
    # Replace each email with anonymized version
    anonymized_text = text
    for username, domain in matches:
        original_email = f"{username}@{domain}"
        # Create a hash of the username (take first 8 chars of hash)
        hashed_username = hashlib.md5(username.encode()).hexdigest()[:8]
        anonymized_email = f"{hashed_username}@{domain}"
        anonymized_text = anonymized_text.replace(original_email, anonymized_email)
    
    return anonymized_text

def anonymize_dataframe(df):
    """
    Anonymize a pandas DataFrame by:
    1. Removing explicit PII columns
    2. Anonymizing emails in all text fields
    
    Args:
        df: pandas DataFrame to anonymize
    
    Returns:
        Anonymized pandas DataFrame
    """
    # Create a copy to avoid modifying the original
    df_anon = df.copy()
    
    # Create user IDs
    df_anon['user_id'] = [f"User_{i+1}" for i in range(len(df_anon))]
    
    # Always anonymize emails in text columns
    text_columns = df_anon.select_dtypes(include=['object']).columns
    for col in text_columns:
        # Anonymize any email address occurrences in the text
        logger.info(f"Anonymizing emails in column: {col}")
        df_anon[col] = df_anon[col].apply(anonymize_email)
    
    # Remove explicit PII columns
    pii_columns = ['email', 'name', 'address', 'phone', 'username']
    for col in pii_columns:
        if col in df_anon.columns:
            df_anon = df_anon.drop(col, axis=1)
    
    return df_anon
