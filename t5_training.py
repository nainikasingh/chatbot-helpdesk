import pandas as pd
import torch
import gc
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

# Load dataset
df = pd.read_csv("t5_training_data.csv").dropna()

# Convert to Hugging Face Dataset format
dataset = Dataset.from_pandas(df)

# Load tokenizer & model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(device)

# Preprocess data
def preprocess_data(examples):
    inputs = ["solve: " + text for text in examples["input"]]
    targets = examples["target"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length", return_tensors="pt").input_ids
    model_inputs["labels"] = labels
    return model_inputs

dataset = dataset.map(preprocess_data, batched=True)

# Free memory
gc.collect()
torch.cuda.empty_cache()

# Fine-tune model
training_args = TrainingArguments(
    output_dir="./fine_tuned_t5/",
    per_device_train_batch_size=3,  # Lower batch size to prevent CUDA OOM
    num_train_epochs=3,
    save_strategy="epoch",
    eval_strategy="no",  # Removed evaluation to prevent missing eval_dataset error
    logging_dir="./logs",
    logging_steps=500,
    fp16=True if torch.cuda.is_available() else False,  # Use FP16 if on GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# Save fine-tuned model
model.save_pretrained("./fine_tuned_t5/")
tokenizer.save_pretrained("./fine_tuned_t5/")
print("Fine-tuned T5 model saved at './fine_tuned_t5/'")
