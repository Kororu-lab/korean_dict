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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import BertProcessing
import gc  # Garbage collector interface

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Hyperparameters
CONTEXT_LENGTH = 30
BATCH_SIZE = 10  # Further reduced batch size to manage memory
REMOVE_DUPLICATES = True
USE_SUBSET = False
KEYWORD = "꼬리"
SIGMA = 1.0
TOKEN_WINDOW = 10
APPLY_SIGMA = False
MAX_LENGTH = 300
SPLIT_LONG_SENTENCES = True
MAX_SENTENCE_LENGTH = 30
SPLIT_AROUND_KEYWORD = 30
USE_BPE = True

# Ensure joblib temporary directory is set
temp_dir = './joblib_temp'
os.makedirs(temp_dir, exist_ok=True)
joblib.parallel.DEFAULT_TEMPORARY_DIRECTORY = temp_dir

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizerFast.from_pretrained('beomi/kcbert-large')
model = BertModel.from_pretrained('beomi/kcbert-large').to(device)

# Load BPE tokenizer
bpe_tokenizer = None
if USE_BPE:
    if not os.path.exists('bpe_tokenizer.json'):
        bpe_tokenizer = Tokenizer(BPE())
        bpe_tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
        bpe_tokenizer.train(files=['./data/preprocessed_corpus.txt'], trainer=trainer)
        bpe_tokenizer.save('bpe_tokenizer.json')
    else:
        bpe_tokenizer = Tokenizer.from_file('bpe_tokenizer.json')
    bpe_tokenizer.post_processor = BertProcessing(("[SEP]", 102), ("[CLS]", 101))

# Load stopwords from file
stopwords_file = './stopwords.txt'
with open(stopwords_file, 'r', encoding='utf-8') as f:
    stop_words = f.read().split()

# Function to remove stopwords
def remove_stopwords(tokens):
    return [token for token in tokens if token not in stop_words]

# Function to split long sentences around keyword
def split_sentence(sentence, keyword, max_length, split_length):
    keywords = [i for i in range(len(sentence)) if sentence.startswith(keyword, i)]
    split_sentences = []
    for index in keywords:
        start = max(0, index - split_length)
        end = min(len(sentence), index + split_length + len(keyword))
        split_sentences.append(sentence[start:end])
    return split_sentences

# Tokenization function
def tokenize(sentences, use_bpe=USE_BPE):
    if use_bpe and bpe_tokenizer:
        return [bpe_tokenizer.encode(sentence).tokens for sentence in sentences]
    else:
        return [bert_tokenizer.tokenize(sentence) for sentence in sentences]

# Function to generate weighted embeddings with TF-IDF
# Function to generate weighted embeddings with TF-IDF
def get_weighted_embeddings(sentences, tfidf_vectorizer, tfidf_matrix, keyword=KEYWORD, sigma=SIGMA, token_window=TOKEN_WINDOW, apply_sigma=APPLY_SIGMA, use_bpe=USE_BPE):
    tokens = tokenize(sentences, use_bpe=use_bpe)
    inputs = bert_tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH, return_offsets_mapping=True)
    offset_mappings = inputs.pop('offset_mapping')
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.cpu().numpy()

    # Calculate weights
    weights = []
    for i, sentence in enumerate(sentences):
        sentence_tokens = tokens[i]
        sentence_tokens = remove_stopwords(sentence_tokens)
        tfidf_weights = np.array([tfidf_matrix[i, tfidf_vectorizer.vocabulary_.get(token, 0)] for token in sentence_tokens if tfidf_vectorizer.vocabulary_.get(token, 0) > 0])
        
        offset_mapping = offset_mappings[i]
        token_length = min(len(inputs['input_ids'][i]), MAX_LENGTH)
        keyword_indices = [j for j, (start, end) in enumerate(offset_mapping) if keyword in sentence[start:end]]
        if keyword_indices:
            keyword_index = keyword_indices[0]  # 첫 번째 "꼬리" 위치
            start_index = max(0, keyword_index - token_window)
            end_index = min(token_length, keyword_index + token_window + 1)
            distances = np.abs(np.arange(start_index, end_index) - keyword_index)
            if apply_sigma:
                distance_weights = np.exp(-distances / (2 * sigma ** 2))
            else:
                distance_weights = np.ones_like(distances)
            full_weight = np.ones(token_length)
            full_weight[start_index:end_index] = distance_weights
            
            # Adjust full_weight length to match tfidf_weights length
            if len(full_weight) > len(tfidf_weights):
                full_weight = full_weight[:len(tfidf_weights)]
            else:
                full_weight = np.pad(full_weight, (0, len(tfidf_weights) - len(full_weight)), 'constant')
            
            full_weight = full_weight * tfidf_weights
            weights.append(full_weight)
        else:
            weights.append(np.ones(token_length))

    # Apply weights
    weighted_embeddings = np.array([emb[:len(w)] * w[:, np.newaxis] for emb, w in zip(embeddings, weights) if len(emb) == len(w)])
    if len(weighted_embeddings) == 0:
        return np.zeros((1, embeddings.shape[2]))  # Avoid empty array
    return weighted_embeddings.mean(axis=1)

# Load and preprocess corpus
logger.info("Loading and preprocessing corpus...")
with open('./data/preprocessed_corpus.txt', 'r', encoding='utf-8') as file:
    sentences = [line.strip() for line in file]

# Filter sentences containing the keyword
sentences_with_tail = [sentence for sentence in sentences if KEYWORD in sentence]
logger.info(f"Found {len(sentences_with_tail)} sentences containing '{KEYWORD}'.")

# Split long sentences if flag is set
if SPLIT_LONG_SENTENCES:
    split_sentences = []
    for sentence in sentences_with_tail:
        if len(sentence) > MAX_SENTENCE_LENGTH:
            split_sentences.extend(split_sentence(sentence, KEYWORD, MAX_SENTENCE_LENGTH, SPLIT_AROUND_KEYWORD))
        else:
            split_sentences.append(sentence)
    sentences_with_tail = split_sentences
    logger.info(f"Split long sentences, resulting in {len(sentences_with_tail)} total sentences.")

# Remove duplicates if flag is set
if REMOVE_DUPLICATES:
    sentences_with_tail = list(set(sentences_with_tail))
    logger.info(f"Reduced to {len(sentences_with_tail)} unique sentences after removing duplicates.")

# Use a smaller subset for debugging if USE_SUBSET is True
if USE_SUBSET:
    sentences_with_tail = sentences_with_tail[:5000]
    logger.info(f"Using a subset of {len(sentences_with_tail)} sentences for debugging.")

# Shuffle sentences to ensure even distribution of processing load
sentences_with_tail = shuffle(sentences_with_tail)

# Fit TF-IDF Vectorizer on the entire corpus
logger.info("Fitting TF-IDF Vectorizer on the entire corpus...")
tfidf_vectorizer = TfidfVectorizer(tokenizer=bert_tokenizer.tokenize, stop_words=stop_words, token_pattern=None)
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences_with_tail)

# Generate weighted embeddings with progress tracking and batch processing
logger.info("Generating weighted embeddings in batches...")
embeddings = []
for i in tqdm(range(0, len(sentences_with_tail), BATCH_SIZE)):
    batch_sentences = sentences_with_tail[i:i+BATCH_SIZE]
    batch_embeddings = get_weighted_embeddings(batch_sentences, tfidf_vectorizer, tfidf_matrix, apply_sigma=APPLY_SIGMA, use_bpe=USE_BPE)
    embeddings.append(batch_embeddings)
    gc.collect()  # Collect garbage after each batch to free up memory
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

