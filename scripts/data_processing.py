import pandas as pd
from scripts.utils import download_preprocess_image
from scripts.ocr import perform_ocr
from torchvision import transforms
from transformers import BertTokenizer

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

def preprocess_image(img, resize=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img

def convert_text_to_tensor(text, tokenizer):
    """
    Convert text to tensor using BERT tokenizer.
    
    Parameters:
    - text (str): The input text to convert.
    - tokenizer: The BERT tokenizer instance.
    
    Returns:
    - torch.Tensor: Tensor representation of the text.
    """
    inputs = tokenizer(text, return_tensors='pt')
    return inputs['input_ids'].flatten()


def extract_and_process_image(csv_path, method='BERT'):
    """
    Extract text from images specified in a CSV file and process it.
    
    Parameters:
    - csv_path (str): Path to the CSV file containing image URLs.
    - method (str): Method to use ('BERT' or 'LLM'). Determines the output format.
    
    Returns:
    - list: List of text tensors if method is 'BERT', otherwise list of plain text strings.
    """
    df = pd.read_csv(csv_path)
    text_outputs = []
    
    if method == 'BERT':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    for _, row in df.iterrows():
        image_url = row.get('image_url_ingredients')
        if image_url:
            text = extract_text_from_image(image_url)
            if method == 'BERT':
                text_tensor = convert_text_to_tensor(text, tokenizer)
                text_outputs.append(text_tensor)
            elif method == 'LLM':
                text_outputs.append(text)
    
    return text_outputs

def extract_text_from_image(image_url: str) -> str:
    """
    Extracts text from an image provided at the given URL.

    Parameters:
    - image_url (Any): The URL of the image from which to extract text.

    Returns:
    - str: The extracted text from the image.
    """
    try:
        # Fetch the image from the URL
        img = download_preprocess_image(image_url)

        # Use pytesseract to perform OCR on the image
        extracted_text = perform_ocr(img)

        return extracted_text

    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""