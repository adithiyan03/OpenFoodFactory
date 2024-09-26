import torch
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

def train_bert_model(texts: list, labels: list) -> BertForTokenClassification:
    """
    Train the BERT model using the provided texts and labels.
    
    Parameters:
    - texts: List of input texts.
    - labels: List of labels corresponding to the texts.
    
    Returns:
    - The trained BERT model.
    """

    class CustomDataset(Dataset):
        def __init__(self, texts, labels, tokenizer):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer.encode_plus(
                self.texts[idx],
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.1, random_state=42)
    
    train_dataset = CustomDataset(X_train, y_train, tokenizer)
    val_dataset = CustomDataset(X_val, y_val, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_loader) * 4
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    for epoch in range(4):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(**{key: val.squeeze(1) for key, val in batch.items()})
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
    
    return model

def load_model() -> tuple:
    """
    Load the trained model weights.
    
    Returns:
    - The BERT model with loaded weights and tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    return model, tokenizer

def predict_entities(text: str, model: BertForTokenClassification, tokenizer: BertTokenizer) -> tuple:
    """
    Predict entities in the provided text using the trained model.
    
    Parameters:
    - text: The input text for entity recognition.
    - model: The BERT model for prediction.
    - tokenizer: The BERT tokenizer.
    
    Returns:
    - Tuple of tokens and predicted labels.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    labels = [model.config.id2label[pred.item()] for pred in predictions[0]]
    
    return tokens, labels
