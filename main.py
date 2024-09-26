import yaml
from scripts.data_preparation import prepare_data_from_csv
from scripts.model_training import train_model
from scripts.ocr import extract_text_from_image
from scripts.validation import predict_entities
from transformers import BertTokenizer
import torch


def main():
    config_path = 'config/config.yaml'
    csv_path = 'data/openfoodfactory.csv'

    # Load configuration
    config = load_config(config_path)
    nutrient_buckets = config['nutrient_buckets']

    # Prepare data from CSV
    texts, labels = prepare_data_from_csv(csv_path, nutrient_buckets)

    # Train the model
    train_bert_model(texts, labels)

    # Extract and process images
    text_tensors = extract_and_process_images(csv_path)

    # Convert list of text tensors to a tensor dataset
    text_tensors = [torch.tensor(tensor, dtype=torch.int32) for tensor in text_tensors]
    tensor_dataset = TensorDataset(*text_tensors)  # TensorDataset does not need labels here

    # Create DataLoader
    dataloader = DataLoader(tensor_dataset, batch_size=4, shuffle=False)

    # Load the trained model
    model, tokenizer = load_model()

    # Predict entities in batches
    model.eval()
    for batch in dataloader:
        batch_texts = [tensor.numpy().astype(str) for tensor in batch]
        
        for text in batch_texts:
            text_str = ''.join(map(chr, text))
            tokens, predicted_labels = predict_entities(text_str, model, tokenizer)
            print(f"Predicted labels for text: {text_str}")
            print(predicted_labels)

if __name__ == '__main__':
    main()
