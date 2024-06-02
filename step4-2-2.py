import numpy as np
import random
import pickle
import torch

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load context vectors and sentences
with open('./results/context_vectors.pkl', 'rb') as f:
    context_vectors = pickle.load(f)

with open('./results/sentences_with_tail.pkl', 'rb') as f:
    sentences_with_tail = pickle.load(f)

# Data structure to store annotations
annotations = {}
unk_indices = set()
annotated_meaning_numbers = {}
last_meaning_number = None

def highlight_keyword(sentence, keyword="꼬리"):
    return sentence.replace(keyword, f"\033[1m{keyword}\033[0m")

def present_example(sentences, annotations, last_meaning_number, unk_indices):
    remaining_indices = [i for i in range(len(sentences)) if i not in annotations and i not in unk_indices]
    if not remaining_indices:
        print("All examples have been annotated.")
        return None, None

    while True:
        random_index = random.choice(remaining_indices)
        if last_meaning_number is None or last_meaning_number not in annotated_meaning_numbers.get(random_index, []):
            break

    example_sentence = sentences[random_index]
    highlighted_sentence = highlight_keyword(example_sentence)
    print(f"Example: {highlighted_sentence}")
    return random_index, example_sentence

def get_user_annotation(index):
    while True:
        meaning_number = input("Enter the meaning number for this example (0 = UNK): ")
        if meaning_number.isdigit():
            meaning_number = int(meaning_number)
            if meaning_number >= 0:
                break
            else:
                print("Meaning number must be a non-negative integer.")
        else:
            print("Please enter a valid number.")
    
    # Skip if the meaning number is 0 (UNK)
    if meaning_number == 0:
        print("Skipping this example as it is marked as UNK.")
        unk_indices.add(index)
        return None, None

    # Automatically fill if the same meaning is already annotated for this example
    if meaning_number in annotated_meaning_numbers.get(index, []):
        meaning_comment = annotations[index][1]  # Use the existing comment
        print(f"This example has already been annotated with the same meaning. Using existing comment: {meaning_comment}")
    else:
        meaning_comment = input("Enter a comment (optional): ")
    
    return meaning_number, meaning_comment

def find_most_different_example(context_vectors, annotated_indices, current_meaning, last_meaning_number, unk_indices):
    if not annotated_indices:
        return random.choice([i for i in range(len(context_vectors)) if i not in annotations and i not in unk_indices])

    annotated_vectors = np.array([context_vectors[i] for i in annotated_indices])
    mean_embedding = np.mean(annotated_vectors, axis=0)
    distances = np.linalg.norm(context_vectors - mean_embedding, axis=1)

    if last_meaning_number is not None:
        same_meaning_indices = [i for i, meanings in annotated_meaning_numbers.items() if last_meaning_number in meanings]
        same_meaning_vectors = np.array([context_vectors[i] for i in same_meaning_indices])
        if same_meaning_vectors.size > 0:
            same_meaning_mean_embedding = np.mean(same_meaning_vectors, axis=0)
            same_meaning_distances = np.linalg.norm(context_vectors - same_meaning_mean_embedding, axis=1)
            distances += same_meaning_distances

    sorted_indices = np.argsort(distances)[::-1]
    for index in sorted_indices:
        if index not in annotations and index not in unk_indices:
            return index
    return None

def update_ui(annotations, sentences):
    print("\nCurrent annotations:")
    for index, (meaning_number, meaning_comment) in annotations.items():
        highlighted_sentence = highlight_keyword(sentences[index])
        print(f"Example {index + 1}: Meaning {meaning_number}, Comment: {meaning_comment}, Sentence: {highlighted_sentence}")
    print()

def main_loop(sentences, context_vectors):
    global last_meaning_number
    while True:
        if not annotations:
            index, example = present_example(sentences, annotations, last_meaning_number, unk_indices)
        else:
            annotated_indices = [i for i, (meaning, _) in annotations.items() if meaning != 0]
            if not annotated_indices:
                print("No annotated examples with meanings.")
                index, example = present_example(sentences, annotations, last_meaning_number, unk_indices)
            else:
                index = find_most_different_example(context_vectors, annotated_indices, None, last_meaning_number, unk_indices)
                if index is None:
                    print("All examples have been annotated.")
                    break
                example = sentences[index]
                highlighted_sentence = highlight_keyword(example)
                print(f"Example: {highlighted_sentence}")

        if example is None:
            break

        meaning_number, meaning_comment = get_user_annotation(index)

        if meaning_number is not None:
            print(f"Annotating example {index} with meaning {meaning_number} and comment '{meaning_comment}'")
            annotations[index] = (meaning_number, meaning_comment)
            if index not in annotated_meaning_numbers:
                annotated_meaning_numbers[index] = []
            annotated_meaning_numbers[index].append(meaning_number)
            last_meaning_number = meaning_number
            update_ui(annotations, sentences)
        else:
            print(f"Skipping example {index} with meaning {meaning_number}")

if __name__ == "__main__":
    main_loop(sentences_with_tail, context_vectors)
