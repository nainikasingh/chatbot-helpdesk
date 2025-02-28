from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
import pandas as pd
import torch
import os
import gc

# Define Device (Use GPU if available, otherwise use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clear GPU Cache
torch.cuda.empty_cache()
gc.collect()

# Load Dataset
file_path = "faiss_triplet_data.csv"
df = pd.read_csv(file_path, dtype=str)

# Convert to Training Examples
def create_examples(df):
    examples = []
    for _, row in df.iterrows():
        examples.append(InputExample(texts=[str(row['anchor']), str(row['positive']), str(row['negative'])]))
    return examples

train_examples = create_examples(df)

# Fix DataLoader Issue: Use `collate_fn=lambda x: x`
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2, collate_fn=lambda x: x)

# Load Pretrained Sentence Transformer Model
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1").to(device)
print(f"Model loaded on {device}")

# Define Loss Function
train_loss = losses.TripletLoss(model)
print("Loss function initialized")

# Fine-Tune Model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100
)

# Save Fine-Tuned Model
output_path = "fine_tuned_sentence_transformer"
os.makedirs(output_path, exist_ok=True)
model.save(output_path)

print(f"Model saved at: {output_path}")
