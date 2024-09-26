import yaml
from scripts.data_preparation import prepare_data_from_csv
from scripts.model_training import train_model
from scripts.ocr import extract_text_from_image
from scripts.validation import predict_entities
from transformers import BertTokenizer
import torch

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Load configuration
    config = load_config()
    nutrient_buckets = config['nutrient_buckets']

    # Prepare data from CSV
    texts, labels = prepare_data_from_csv('data/openfoodfactory.csv', nutrient_buckets)

    # Train the model
    train_model(texts, labels)

    # Extract text from image
    image_path = 'path/to/your/image.png'
    extracted_text = extract_text_from_image(image_path)

    # Load trained model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load('model.pt'))

    # Predict entities in the extracted text
    tokens, predicted_labels = predict_entities(extracted_text, model, tokenizer)

    # Print results
    print("Tokens:", tokens)
    print("Predicted Labels:", predicted_labels)

if __name__ == '__main__':
    main()
