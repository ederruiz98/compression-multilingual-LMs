from datasets import load_dataset
import random
import re
import sys

lang_code_1 = sys.argv[1]
lang_code_2 = sys.argv[2]

# Language code to full name mapping
language_names = {
    "es": "spanish",
    "en": "english",
    "de": "german",
    "fr": "french",
    "el": "greek",
    "it": "italian",
    "pt": "portuguese",
    "ro": "romanian",
    "bg": "bulgarian",
    "cs": "czech",
    "da": "danish",
    "et": "estonian",
    "fi": "finnish",
    "hu": "hungarian",
    "lt": "lithuanian",
    "lv": "latvian",
    "nl": "dutch",
    "pl": "polish",
    "sk": "slovak",
    "sl": "slovenian",
    "sv": "swedish"
}

# Load the specified pair of languages subset of the europarl dataset
dataset = load_dataset("Helsinki-NLP/europarl", f"{lang_code_1}-{lang_code_2}", split="train")

# Extract all language pairs
dataset = dataset['translation']

# Randomly sample 20,000 sentences from the dataset (around 10k lines should be enough to compute ID)

if len(dataset) < 20000: # Ensure there are at least 20K sentences
    raise ValueError("Dataset does not have enough sentences.")

random.seed(23)  # Set a seed for replicability
sampled_dataset = random.sample(dataset, 20000) # Extract 20K random pairs of sentences

# Extract sentences for both languages
sentences_lang_1 = [sentence[lang_code_1] for sentence in sampled_dataset]
sentences_lang_2 = [sentence[lang_code_2] for sentence in sampled_dataset]

# Function to process sentences into 20-word lines
def process_sentences(sentences):
    # Add a dot at the end of sentences if there's not one already (I saw some sentences with no dot at the end. This would have an impact when the 20-word lines are feeded to the model)
    sentences = [sentence.strip() + '.'
                 if re.search(r'\w$', sentence.strip()) # This checks if the stripped sentence ends with a word character (we don't want to add a dot if the sentence ends with an exclamation or question mark, colon, etc.)
                 else sentence.strip() # Otherwise, keep the sentence unchanged
                 for sentence in sentences]

    # Concatenate all sentences into a single string
    concatenated_sentences = ' '.join(sentences)

    # Split the concatenated sentences into words
    words = concatenated_sentences.split()

    # Initialize variables
    lines = []  # A list to store the lines with 20 words each
    line = ""  # A string to store the current line

    # Iterate over words and create lines with 20 words each
    for word in words: # For loop to iterate over the words in the 'words' list
        if len(line.split()) < 20: # If the current line has less than 20 words, add the word to the line with a space after it
            line += word + " "
        else: # When the current line reaches 20 words, add the line to the 'lines' list and start a new line with the current word
            line = line.rstrip() # Remove space at the end of the line
            if line.endswith('.'): # If the line ends with a dot, do not add a space at the beginning of the next line
                lines.append(line) # Append the current line to 'lines'
                line = word + " " # Start a new line with the current word and a space after it
            else: # If the line does not end with a dot (meaning the sentence is not over), add a space at the beginning of the new line
                lines.append(line) # Append the current line to 'lines'
                line = " " + word + " " # Start a new line with the current word and add a space at the beginning of the new line

    # Append the last line to the 'lines' list after the loop finishes in case the 'line' variable is not empty
    if line:
        lines.append(line)

    return lines

# Process sentences for both languages
lines_lang_1 = process_sentences(sentences_lang_1)
lines_lang_2 = process_sentences(sentences_lang_2)

# Function to generate the filename
def generate_filename(lang_code, subset, part): # subset refers to the subset of the europarl dataset, for instance, "en-es". "Part" refers to the first or second half.
    return f"{part}_{subset}_{language_names[lang_code]}_europarl.txt"

# Function to split the dataset in two (to avoid memory issues) and write the 4 datasets (2 per language) with 20-word lines to text files
def split_and_write(lines, lang_code, subset):
    half = len(lines) // 2
    first_half = lines[:half]
    second_half = lines[half:]

    with open(generate_filename(lang_code, subset, "1"), 'w', encoding='utf-8') as file:
        for line in first_half:
            file.write(line + '\n')

    with open(generate_filename(lang_code, subset, "2"), 'w', encoding='utf-8') as file:
        for line in second_half:
            file.write(line + '\n')

# Split and write datasets with 20-word lines to text files
subset_name = f"{lang_code_1}-{lang_code_2}"
split_and_write(lines_lang_1, lang_code_1, subset_name)
split_and_write(lines_lang_2, lang_code_2, subset_name)


print(f"Datasets have been written to {generate_filename(lang_code_1, subset_name, '1')}, {generate_filename(lang_code_1, subset_name, '2')}, {generate_filename(lang_code_2, subset_name, '1')}, and {generate_filename(lang_code_2, subset_name, '2')} respectively.")


# python datasets_preprocessing.py en es

