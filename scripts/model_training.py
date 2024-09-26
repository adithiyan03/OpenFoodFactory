import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments

class NutrientDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, return_tensors="pt")
        label_ids = self.labels[idx]
        return {**tokens, 'labels': torch.tensor(label_ids)}

def train_model(train_texts: List[str], train_labels: List[List[int]], model_path='model.pt'):
    """
    Train the BERT model.
    
    Parameters:
    - train_texts: List of training texts.
    - train_labels: List of corresponding labels.
    - model_path: Path to save the model weights.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = NutrientDataset(train_texts, train_labels, tokenizer)
    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=2)  # Adjust num_labels if needed

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()
    torch.save(model.state_dict(), model_path)  # Save model weights
