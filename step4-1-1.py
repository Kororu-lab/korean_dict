import numpy as np
from transformers import BertModel, BertTokenizer
import torch
from tqdm import tqdm
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Hyperparameters
CONTEXT_LENGTH = 30
BATCH_SIZE = 100
N_CLUSTERS = 8  # num_k

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
with open('./results/method_embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

# Reduce dimensionality using PCA
print("Reducing dimensionality using PCA...")
pca = PCA(n_components=10)  # Further reduce to 10 dimensions
reduced_embeddings = pca.fit_transform(embeddings)

# Clustering with K-Means
print(f"Clustering with k={N_CLUSTERS}...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(reduced_embeddings)
labels = kmeans.labels_

# Save model and results
print("Saving model and results...")
with open('./results/method_kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
with open('./results/method_labels.pkl', 'wb') as f:
    pickle.dump(labels, f)
with open('./results/method_sentences.pkl', 'wb') as f:
    pickle.dump(sentences_with_tail, f)

# Extract sample sentences
sample_sentences = {i: [] for i in range(N_CLUSTERS)}
for i in range(N_CLUSTERS):
    for idx in np.where(labels == i)[0][:10]:
        sentence = sentences_with_tail[idx]
        keyword_idx = sentence.index("꼬리")
        forward_context = sentence[:keyword_idx][-CONTEXT_LENGTH:]
        backward_context = sentence[keyword_idx+2:][:CONTEXT_LENGTH]
        sample_sentences[i].append(
            f"({forward_context})꼬리({backward_context})"
        )

# Save sample sentences to file
with open('./results/method_sample_sentences.txt', 'w', encoding='utf-8') as f:
    for cluster, sentences in sample_sentences.items():
        f.write(f"Cluster {cluster}:\n")
        for sentence in sentences:
            f.write(sentence + "\n")
        f.write("\n")

print("Sample sentences per cluster saved to ./results/method_sample_sentences.txt")
