# compression-multilingual-LMs

This is the code used for my study titled "Exploring Geometric Compression Across Languages in Multilingual Language Models".

Abstract

This study explores geometric compression of linguistic data across languages in multilingual language models using the Europarl corpus, focusing on three models: BLOOM, XLM-RoBERTa, and Mistral. We estimate the intrinsic dimension (ID) of hidden representations at each layer to quantify geometric compression. In Transformer-based LMs, the last hidden representation arises from a series of intermediate representations computed through a number of identical modules. Our analysis reveals that the ID of these representations is significantly smaller than the ambient dimension, with distinct compression patterns across languages. Languages from the same family exhibit similar ID amplitudes, suggesting that shared linguistic properties impact the dimensionality of model representations. Additionally, we find that the model’s performance on a language correlates with the ID amplitude at the first high-dimensionality phase, indicating that the learned linguistic properties influence compression. These findings complement those found in other studies, bringing new insights to our understanding of how state-of-the-art LLMs process and compress linguistic data in different languages.

How to run this code:

1. datasets_preprocessing.py -> This processes the given subset of the Europarl corpus, extracting 20k sentences for each language and converting them into 20-word lines. Then both datasets are split in 2 (to avoid memory issues) and the 4 datasets (2 per language) are written to text files. Usage example:

  python datasets_preprocessing.py en es
(This processes the "en-es" subset and outputs 4 text files: 1_en-es_english_europarl.txt, 1_en-es_spanish_europarl.txt, 2_en-es_english_europarl.txt, 2_en-es_spanish_europarl.txt).

3. extract_final_representations.py (developed by Emily Cheng and adapted by Marco Baroni) -> This extracts the hidden representation at each transformer layer. Usage example:

  python extract_final_representations.py bigscience/bloom-3b 8 1_en-es_english_europarl.txt 1_bloom-3b_en-es_english_europarl_residual
(The first argument is the model, the second is the batch size, the third is the dataset, and the fourth is the out pickle file. For each language pair this code needs to be ran 4 times, one for each of the 4 datasets from datasets_preprocessing.py).

5. multi_input_get_by_layer_ids.py (developed by Emily Cheng and adapted by Marco Baroni) -> This computes the intrinsic dimension of the extracted hidden representation. Usage example:

  python multi_input_get_by_layer_ids.py bloom-3b_en-es_english_europarl_residual 2 MLE bloom-3b_en-es_english_europarl_MLE_id
(First argument is the output pickle file from extract_final_representations.py, second is the number of splits with that name, third is the ID estimator, and fourth is the out pickle file. This snippet needs to be ran twice, once for each language in the subset).
