from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch

class NutrientDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer: BertTokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        # Align labels with tokenized input
        labels = [0] * self.max_len
        for i, token in enumerate(text.split()):
            labels[i] = label.split()[i] if i < len(label.split()) else 0
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# Setup model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Adjust num_labels

# Prepare data
texts, labels = prepare_data_for_bert(text_data, nutrient_buckets)
dataset = NutrientDataset(texts, labels, tokenizer, max_len=128)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()
