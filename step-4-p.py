import re
from tqdm import tqdm

# Function to clean and preprocess the text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

# Load and preprocess corpus
print("Loading and preprocessing corpus...")
corpus_file = './data/merged_corpus.txt'
preprocessed_sentences = []

with open(corpus_file, 'r', encoding='utf-8') as file:
    for line in tqdm(file):
        sentences = line.split('.')  # 문장 단위로 분리
        for sentence in sentences:
            cleaned_sentence = preprocess_text(sentence)
            if cleaned_sentence:
                preprocessed_sentences.append(cleaned_sentence)

# Save preprocessed corpus to a file
preprocessed_corpus_file = './data/preprocessed_corpus.txt'
with open(preprocessed_corpus_file, 'w', encoding='utf-8') as file:
    for sentence in preprocessed_sentences:
        file.write(sentence + '\n')

print(f"Preprocessed corpus saved to {preprocessed_corpus_file}")
