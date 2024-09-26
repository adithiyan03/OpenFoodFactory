import torch
from transformers import BertTokenizer, BertForTokenClassification
from typing import Tuple, List

def predict_entities(text: str, model, tokenizer) -> Tuple[List[str], List[int]]:
    """
    Predict entities in the text using the trained BERT model.
    
    Parameters:
    - text: Text to perform NER on.
    - model: Trained BERT model.
    - tokenizer: BERT tokenizer.
    
    Returns:
    - Tuple of tokens and predicted labels.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
    
    predictions = torch.argmax(outputs.logits, dim=2)
    predicted_labels = predictions[0].numpy().tolist()
    tokens = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])

    return tokens, predicted_labels
