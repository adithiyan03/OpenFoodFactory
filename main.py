from scripts.preprocessing import preprocess_image
from scripts.ocr import perform_ocr
from scripts.data_preparation import prepare_data_for_bert
from scripts.model_training import train_model
from scripts.validation import evaluate_model

def main():
    # Step 1: Preprocess image
    preprocessed_image = preprocess_image('path_to_image')
    
    # Step 2: Perform OCR
    text_data = perform_ocr(preprocessed_image)
    
    # Step 3: Prepare data for BERT
    texts, labels = prepare_data_for_bert(text_data, nutrient_buckets)
    
    # Step 4: Train the model
    model, tokenizer = train_model(texts, labels)
    
    # Step 5: Evaluate the model
    evaluate_model(model, tokenizer)
    
if __name__ == "__main__":
    main()

