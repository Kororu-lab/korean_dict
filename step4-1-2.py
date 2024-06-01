import numpy as np
from transformers import BertModel, BertTokenizer
import torch
from tqdm import tqdm
import pickle
from sklearn.cluster import MiniBatchKMeans

# Hyperparameters
CONTEXT_LENGTH = 30
BATCH_SIZE = 100
N_CLUSTERS = 8 # num_k

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

# Load saved embeddings and sentences
with open('./results/method_embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)
with open('./results/method_sentences.pkl', 'rb') as f:
    sentences_with_tail = pickle.load(f)

# Iterative refinement
print("Refining embeddings iteratively...")
for iteration in tqdm(range(10)):  # Number of iterations
    new_embeddings = []
    for i in range(0, len(sentences_with_tail), BATCH_SIZE):
        batch_sentences = sentences_with_tail[i:i+BATCH_SIZE]
        
        # Ensure that the sentences are properly batched
        batch_embeddings = []
        for sentence in batch_sentences:
            embedding = get_embeddings([sentence])
            batch_embeddings.append(embedding[0])
        
        new_embeddings.append(np.array(batch_embeddings))
    
    new_embeddings = np.vstack(new_embeddings)
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=0, batch_size=BATCH_SIZE).fit(new_embeddings)
    labels = kmeans.labels_

# Save final model and results
print("Saving final model and results...")
with open('./results/final_kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
with open('./results/final_labels.pkl', 'wb') as f:
    pickle.dump(labels, f)
with open('./results/final_sentences.pkl', 'wb') as f:
    pickle.dump(sentences_with_tail, f)

# Extract final sample sentences
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

# Save final sample sentences to file
with open('./results/final_sample_sentences.txt', 'w', encoding='utf-8') as f:
    for cluster, sentences in sample_sentences.items():
        f.write(f"Cluster {cluster}:\n")
        for sentence in sentences:
            f.write(sentence + "\n")
        f.write("\n")

print("Final sample sentences per sense saved to ./results/final_sample_sentences.txt")
