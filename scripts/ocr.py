import numpy as np
from pytesseract import image_to_string

def perform_ocr(image: np.ndarray) -> str:
    """
    Extract text from the preprocessed image using OCR.
    
    Parameters:
    - image: Preprocessed image as a NumPy array.
    
    Returns:
    - Extracted text as a string.
    """
    # Perform OCR
    text = image_to_string(image, lang='eng')
    
    return text
