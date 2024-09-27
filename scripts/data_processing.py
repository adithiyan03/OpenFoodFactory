import pandas as pd
import numpy as np
import torch
from ocr import extract_text_from_image
from scripts.utils import download_preprocess_image

def map_column_to_label(column: str, nutrient_buckets: dict) -> str:
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
                return f"B-{nutrient.replace(' ', '-').upper()}"
    return "O"

def prepare_data_from_csv(csv_path: str, nutrient_buckets: dict) -> tuple:
    """
    Prepare data for BERT model.
    
    Parameters:
    - df: DataFrame containing nutritional data.
    - nutrient_buckets: Dictionary for mapping columns to labels.
    
    Returns:
    - Tuple of input texts and corresponding labels.
    """
    X = []  # Input tokens
    Y = []  # Corresponding labels
    df = pd.read_csv(csv_path)
    df = df.loc[:,'energy-kj_100g':]
    
    for index, row in df.iterrows():
        tokens = []
        labels = []
        for column in df.columns:
            value = row[column]
            if pd.notna(value):  # Only include non-NaN values
                token = f"{column} {value}"
                label = map_column_to_label(column, nutrient_buckets)
                
                tokens.append(token)
                labels.append(label)
        
        if tokens:  # Only add non-empty sequences
            X.append(' '.join(tokens))
            Y.append(' '.join(labels))
    
    return X, Y

def extract_and_process_images(csv_path: str) -> list:
    """
    Read image URLs from the CSV, download images, perform OCR, and convert text to tensor.
    
    Parameters:
    - csv_path: Path to the CSV file containing image URLs.
    
    Returns:
    - List of tensors representing extracted text.
    """
    df = pd.read_csv(csv_path)
    text_tensors = []
    
    for index, row in df.iterrows():
        image_url = row.get('image_nutrition_url')
        if pd.notna(image_url):
            try:
                image = download_preprocess_image(image_url)
                extracted_text = extract_text_from_image(image)
                text_tensor = torch.tensor(list(map(ord, extracted_text)), dtype=torch.int32)
                text_tensors.append(text_tensor)
                
            except Exception as e:
                print(f"Error processing image at index {index}: {e}")
    
    return text_tensors
