import os
import numpy as np
from transformers import BertModel, BertTokenizerFast
import torch
from tqdm import tqdm
import pickle
from sklearn.decomposition import PCA
import hdbscan
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Hyperparameters
CONTEXT_LENGTH = 30
BATCH_SIZE = 25  # Adjusted batch size to manage memory
REMOVE_DUPLICATES = True  # Set to False to keep duplicates
USE_SUBSET = False  # Set to True to use a smaller subset for debugging
KEYWORD = "꼬리"
SIGMA = 1.0  # Gaussian weighting parameter
TOKEN_WINDOW = 15  # Consider only this range of tokens around the keyword

# Ensure joblib temporary directory is set
temp_dir = './joblib_temp'
os.makedirs(temp_dir, exist_ok=True)
joblib.parallel.DEFAULT_TEMPORARY_DIRECTORY = temp_dir

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load BERT model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained('beomi/kcbert-large')
model = BertModel.from_pretrained('beomi/kcbert-large').to(device)

# Function to generate weighted embeddings
def get_weighted_embeddings(sentences, keyword=KEYWORD, sigma=SIGMA, token_window=TOKEN_WINDOW):
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, return_offsets_mapping=True)
    offset_mappings = inputs.pop('offset_mapping')
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.cpu().numpy()
    
    # Calculate weights
    weights = []
    for i, sentence in enumerate(sentences):
        offset_mapping = offset_mappings[i]
        token_length = len(inputs['input_ids'][i])
        keyword_indices = [j for j, (start, end) in enumerate(offset_mapping) if keyword in sentence[start:end]]
        if keyword_indices:
            keyword_index = keyword_indices[0]  # Use the first occurrence of the keyword
            start_index = max(0, keyword_index - token_window)
            end_index = min(token_length, keyword_index + token_window + 1)
            distances = np.abs(np.arange(start_index, end_index) - keyword_index)
            weight = np.exp(-distances / (2 * sigma ** 2))
            full_weight = np.ones(token_length)
            full_weight[start_index:end_index] = weight
            weights.append(full_weight)
        else:
            weights.append(np.ones(token_length))
    
    # Apply weights
    weighted_embeddings = np.array([emb[:len(w)] * w[:, np.newaxis] for emb, w in zip(embeddings, weights)])
    return weighted_embeddings.mean(axis=1)

# Load and preprocess corpus
logger.info("Loading and preprocessing corpus...")
with open('./data/preprocessed_corpus.txt', 'r', encoding='utf-8') as file:
    sentences = [line.strip() for line in file]

# Filter sentences containing the keyword
sentences_with_tail = [sentence for sentence in sentences if KEYWORD in sentence]
logger.info(f"Found {len(sentences_with_tail)} sentences containing '{KEYWORD}'.")

# Remove duplicates if flag is set
if REMOVE_DUPLICATES:
    sentences_with_tail = list(set(sentences_with_tail))
    logger.info(f"Reduced to {len(sentences_with_tail)} unique sentences after removing duplicates.")

# Use a smaller subset for debugging if USE_SUBSET is True
if USE_SUBSET:
    sentences_with_tail = sentences_with_tail[:5000]
    logger.info(f"Using a subset of {len(sentences_with_tail)} sentences for debugging.")

# Generate weighted embeddings with progress tracking and batch processing
logger.info("Generating weighted embeddings in batches...")
embeddings = []
for i in tqdm(range(0, len(sentences_with_tail), BATCH_SIZE)):
    batch_sentences = sentences_with_tail[i:i+BATCH_SIZE]
    batch_embeddings = get_weighted_embeddings(batch_sentences)
    embeddings.append(batch_embeddings)
embeddings = np.vstack(embeddings)

# Save embeddings
with open('./results/method_embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

# Reduce dimensionality using PCA
logger.info("Reducing dimensionality using PCA...")
pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(embeddings)

# Display explained variance ratio
explained_variance = pca.explained_variance_ratio_
logger.info(f"Explained variance ratio: {explained_variance}")
logger.info(f"Cumulative explained variance: {np.cumsum(explained_variance)}")

# Check for NaNs or infinite values
if np.isnan(reduced_embeddings).any() or np.isinf(reduced_embeddings).any():
    logger.warning("Reduced embeddings contain NaNs or infinite values.")
    reduced_embeddings = np.nan_to_num(reduced_embeddings)

# Clustering with HDBSCAN
logger.info("Clustering with HDBSCAN...")
try:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=10)
    labels = clusterer.fit_predict(reduced_embeddings)
except Exception as e:
    logger.error(f"Error during clustering: {e}")
    labels = np.array([-1] * len(reduced_embeddings))  # Mark all points as noise

# Save model and results
logger.info("Saving model and results...")
with open('./results/method_hdbscan.pkl', 'wb') as f:
    pickle.dump(clusterer, f)
with open('./results/method_labels.pkl', 'wb') as f:
    pickle.dump(labels, f)
with open('./results/method_sentences.pkl', 'wb') as f:
    pickle.dump(sentences_with_tail, f)

# Extract sample sentences
unique_labels = np.unique(labels)
sample_sentences = {i: [] for i in unique_labels}
for i in unique_labels:
    for idx in np.where(labels == i)[0][:10]:
        sentence = sentences_with_tail[idx]
        keyword_idx = sentence.index(KEYWORD)
        forward_context = sentence[:keyword_idx][-CONTEXT_LENGTH:]
        backward_context = sentence[keyword_idx+len(KEYWORD):][:CONTEXT_LENGTH]
        sample_sentences[i].append(
            f"({forward_context}){KEYWORD}({backward_context})"
        )

# Save sample sentences to file
with open('./results/method_sample_sentences.txt', 'w', encoding='utf-8') as f:
    for cluster, sentences in sample_sentences.items():
        f.write(f"Cluster {cluster}:\n")
        for sentence in sentences:
            f.write(sentence + "\n")
        f.write("\n")

logger.info("Sample sentences per cluster saved to ./results/method_sample_sentences.txt")

# Clean up temporary files
import shutil
shutil.rmtree(temp_dir)
