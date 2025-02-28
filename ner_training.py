import spacy
import json
from spacy.training import Example
from spacy.tokens import DocBin

# Load existing blank model
nlp = spacy.blank("en")

# Create NER pipeline if not present
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Load training data
with open("ner_training_data.json", "r", encoding="utf-8") as f:
    training_data = json.load(f)

# Add labels to the NER pipeline
for entry in training_data:
    for ent in entry["entities"]:
        ner.add_label(ent["label"])

# Convert training data to spaCy format
db = DocBin()

# Start training
nlp.begin_training()

for entry in training_data:
    doc = nlp.make_doc(entry["sentence"])  # Tokenize the sentence
    ents = []

    for ent in entry["entities"]:
        start_char = entry["sentence"].find(ent["text"])  # Find entity start position
        end_char = start_char + len(ent["text"])  # Calculate end position

        if start_char == -1:
            print(f"⚠️ Warning: Entity '{ent['text']}' not found in '{entry['sentence']}'")
            continue

        span = doc.char_span(start_char, end_char, label=ent["label"])  # Create entity span

        if span is not None:
            ents.append(span)
        else:
            print(f"Warning: Could not create span for '{ent['text']}' in '{entry['sentence']}'")

    doc.ents = ents  # Assign extracted entities to the doc

    # Convert to spaCy Example format
    example = Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in ents]})
    db.add(doc)

# Save processed data for training
db.to_disk("train.spacy")

# Create config file for training
config_text = """
[paths]
train = "train.spacy"
dev = "train.spacy"

[system]
gpu_allocator = "pytorch"

[nlp]
lang = "en"
pipeline = ["ner"]

[components]
[components.ner]
factory = "ner"

[training]
seed = 42 
dropout = 0.2
max_steps = 1000
"""
with open("config.cfg", "w") as f:
    f.write(config_text)

# Train spaCy NER model
import os
os.system("python -m spacy train config.cfg --output ./spacy_model --gpu-id 0")

print("NER model trained and saved in ./spacy_model/")
