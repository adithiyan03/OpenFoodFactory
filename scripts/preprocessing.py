from PIL import Image
from torchvision import transforms
import numpy as np

def preprocess_image(image_path: str, resize: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess the image for better OCR results.
    Resize and normalize the image.

    Parameters:
    - image_path: Path to the image file.
    - resize: Tuple specifying the dimensions to resize the image to (width, height).

    Returns:
    - Preprocessed image as a NumPy array.
    """
    img = Image.open(image_path).convert('RGB')  # Convert to RGB if not already
    
    # Resize image to desired dimensions
    img = img.resize(resize)
    
    # Normalize image using torchvision transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet-like normalization
    ])
    
    img = transform(img)
    return img
