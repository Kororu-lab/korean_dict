import numpy as np
from transformers import BertModel, BertTokenizer
import torch
from tqdm import tqdm
import pickle

# Hyperparameters
BATCH_SIZE = 100

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-large')
model = BertModel.from_pretrained('beomi/kcbert-large').to(device)

# Function to generate embeddings
def get_embeddings(sentences):
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# Load and preprocess corpus
print("Loading and preprocessing corpus...")
with open('./data/preprocessed_corpus.txt', 'r', encoding='utf-8') as file:
    sentences = [line.strip() for line in file]

# Filter sentences containing "꼬리"
sentences_with_tail = [sentence for sentence in sentences if "꼬리" in sentence]
print(f"Found {len(sentences_with_tail)} sentences containing '꼬리'.")

# Generate embeddings with progress tracking and batch processing
print("Generating embeddings in batches...")
embeddings = []
for i in tqdm(range(0, len(sentences_with_tail), BATCH_SIZE)):
    batch_sentences = sentences_with_tail[i:i+BATCH_SIZE]
    batch_embeddings = get_embeddings(batch_sentences)
    embeddings.append(batch_embeddings)
embeddings = np.vstack(embeddings)

# Save embeddings
with open('./results/context_vectors.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

with open('./results/sentences_with_tail.pkl', 'wb') as f:
    pickle.dump(sentences_with_tail, f)

print("Context vectors and sentences saved.")
