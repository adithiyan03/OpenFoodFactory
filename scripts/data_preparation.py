import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

def map_column_to_label(column: str, nutrient_buckets: Dict) -> str:
    """
    Map column name to label.
    
    Parameters:
    - column: Column name or token.
    - nutrient_buckets: Dictionary for mapping columns to labels.
    
    Returns:
    - Label as a string.
    """
    for category, nutrients in nutrient_buckets.items():
        for nutrient, columns in nutrients.items():
            if column in columns:
                return f"B-{nutrient.replace(' ', '-').upper()}"  # Example: B-VITAMINS
    return "O"  # Use "O" for non-nutrient labels

def prepare_data_from_csv(csv_file: str, nutrient_buckets: Dict) -> Tuple[List[str], List[List[int]]]:
    """
    Prepare data for BERT model from CSV.
    
    Parameters:
    - csv_file: Path to the CSV file.
    - nutrient_buckets: Dictionary for mapping columns to labels.
    
    Returns:
    - Tuple of input strings and corresponding labels.
    """
    df = pd.read_csv(csv_file)
    texts, labels = [], []

    for index, row in df.iterrows():
        tokens = []
        label_ids = []
        
        for column in df.columns:
            value = row[column]
            if pd.notna(value):  # Only include non-NaN values
                token = f"{column} {value}"
                label = map_column_to_label(column, nutrient_buckets)
                
                tokens.append(token)
                label_ids.append(label)
        
        if tokens:
            texts.append(' '.join(tokens))
            labels.append(label_ids)

    # Pad labels to ensure all sequences are of the same length
    max_len = max(len(label) for label in labels)
    labels = [label + [0] * (max_len - len(label)) for label in labels]  # Padding with 0 for non-nutrient labels

    return texts, labels
