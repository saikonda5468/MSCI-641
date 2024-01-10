import numpy as np
import pandas as pd
import gensim.models
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
import os
import sys
import pickle

# Load pre-trained word2vec embeddings
word2vec_model_path = "C:/Users/saiko/OneDrive/Desktop/641/Assig3/word2vec_model.bin"  # Update with your word2vec model path
word2vec_model = gensim.models.Word2Vec.load(word2vec_model_path)

# Load the saved model
model_name = sys.argv[2]  # Update with the path to your saved model file
model_file_path = f"C:/Users/saiko/OneDrive/Desktop/641/Assig4/nn_{model_name}.pkl"
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

def convert_csv_to_dataframe(data_file_path):
    data = []
    with open(data_file_path, 'r') as file:
        for line in file:
            data.append(line)

    df = pd.DataFrame({'text': data})
    return df

def preprocess_data(dataframe, tokenizer, word2vec_model):
    text_data = dataframe['text']
    tokenized_data = []
    for text in text_data:
        tokens = tokenizer(text)
        tokenized_data.append(tokens)
    word_embeddings = []
    for doc in tokenized_data:
        embeddings = []
        for token in doc:
            if token in word2vec_model.wv.key_to_index:
                embeddings.append(word2vec_model.wv[token])
        if embeddings:
            embeddings = np.array(embeddings)
            doc_embedding = np.mean(embeddings, axis=0)
        else:
            doc_embedding = np.zeros(word2vec_model.vector_size)
        word_embeddings.append(doc_embedding)

    word_embeddings = np.array(word_embeddings)
    return word_embeddings

# Preprocess the data
tokenizer = CountVectorizer().build_tokenizer()
file_path = sys.argv[1]  # Update with the path to your test data file
test_data = convert_csv_to_dataframe(file_path)
test_data_embedding = preprocess_data(test_data, tokenizer, word2vec_model)

#Model prediction

Prediction = model.predict(test_data_embedding)
predicted_labels = np.argmax(Prediction, axis=1)
print(predicted_labels)

# Print predicted labels for each instance
for i in range(len(predicted_labels)):
    print(f"Text: {test_data['text'][i]}, Predicted Label: {predicted_labels[i]}")
actual_labels = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])  # Update with your actual labels
accuracy = accuracy_score(actual_labels, predicted_labels)
print("Accuracy:", accuracy)