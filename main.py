import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
from scripts.data_processing import extract_and_process_image
from scripts.data_processing import prepare_data_from_csv
from scripts.utils import load_config
from scripts.llm_processing import call_llm
from scripts.model_training import train_bert_model, load_model
from scripts.validation import predict_entities


def main():
    config_path = 'config/config.yaml'
    csv_path = 'data/openfoodfactory.csv'
    method = config.get('method', 'LLM') 

    # Load configuration
    config = load_config(config_path)
    nutrient_buckets = config['nutrient_buckets']

    # Prepare data from CSV
    texts, labels = prepare_data_from_csv(csv_path, nutrient_buckets)
    

    if method == 'BERT':
            
            # Train the model
            train_bert_model(texts, labels)

            # Extract and process images
            text_tensors = extract_and_process_image(csv_path)

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

    elif method == 'LLM':
        # Extract and process images
        text_tensors = extract_and_process_image(csv_path)

        # Convert list of text tensors to a list of strings
        texts = [''.join(map(chr, tensor.numpy())) for tensor in text_tensors]

        # Perform LLM-based NER and classification
        llm_results = call_llm(texts)
        print("LLM Results:")
        print(llm_results)

if __name__ == '__main__':
    main()
