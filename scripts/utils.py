import yaml
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.
    
    Parameters:
    - config_path: Path to the YAML configuration file.
    
    Returns:
    - Dictionary with configuration data.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def download_preprocess_image(url: str):
    """
    Download an image from a URL.
    
    Parameters:
    - url: URL of the image to download.
    
    Returns:
    - Image object.
    """
    
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    img = image.resize(resize)
    
    # Normalize image using torchvision transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet-like normalization
    ])
    
    img = transform(img)
    return img

