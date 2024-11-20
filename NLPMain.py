import pandas as pd
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

folder_path = 'D:/College/Computer Science/NLP/Final Project/Center Data/Center Data'  # Update with your file path
df = pd.DataFrame()

data = []  # List to store data temporarily

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            data.append({'label': 1, 'content': text})

# Convert list of dictionaries to a DataFrame
df = pd.DataFrame(data)

print(df.head())

train_df, test_df = train_test_split(df, test_size=0.2, random_state=4)

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['text']
        label = self.data.iloc[index]['label']
        
        # Tokenize and encode
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['label'].unique()))


BATCH_SIZE = 16
MAX_LEN = 128

# Create training and validation datasets
train_dataset = TextDataset(train_df, tokenizer, MAX_LEN)
test_dataset = TextDataset(test_df, tokenizer, MAX_LEN)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10
)

# Define metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)
trainer.train()
eval_result = trainer.evaluate()
print(eval_result)
# def predict(text, tokenizer, model, max_len=128):
#     encoding = tokenizer.encode_plus(
#         text,
#         add_special_tokens=True,
#         max_length=max_len,
#         padding='max_length',
#         truncation=True,
#         return_tensors='pt'
#     )
#     input_ids = encoding['input_ids']
#     attention_mask = encoding['attention_mask']
    
#     output = model(input_ids, attention_mask=attention_mask)
#     prediction = torch.argmax(output.logits, dim=1)
#     return prediction.item()

# # Example prediction
# text = "Your sample text here"
# print("Prediction:", predict(text, tokenizer, model))
