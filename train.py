import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import torch

# Load the dataset
df = pd.read_csv('/content/grouped_data.csv')

# Preprocess the data
df['reviewText'] = df['reviewText'].apply(lambda x: x.strip())
df['asin'] = df['asin'].apply(lambda x: x.strip())

# Split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Create a custom dataset class
class RecommendationDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        review_text = self.df.iloc[idx]['reviewText']
        asin = self.df.iloc[idx]['asin']

        # Preprocess the text using the tokenizer
        inputs = self.tokenizer.encode_plus(
            review_text,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Create a label tensor (e.g., 1 if the asin is in the list of recommended products, 0 otherwise)
        labels = torch.tensor([1 if asin in self.df['asin'].values else 0])

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': labels
        }

# Create a dataset instance for training and validation
tokenizer = AutoTokenizer.from_pretrained('t5-small')
model = AutoModelForSequenceClassification.from_pretrained('t5-small', num_labels=2)

train_dataset = RecommendationDataset(train_df, tokenizer)
val_dataset = RecommendationDataset(val_df, tokenizer)

# Create a data loader for training and validation
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Define the training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    evaluation_strategy='epoch',
    save_strategy='epoch',  # Update this to match the eval strategy
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    gradient_accumulation_steps=4,
    greater_is_better=True,
    fp16=True,
    save_total_limit=2,
    no_cuda=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=lambda pred: {'accuracy': accuracy_score(pred.label_ids, pred.predictions.argmax(-1))}
)
# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()