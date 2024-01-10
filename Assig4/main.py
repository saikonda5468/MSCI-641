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

def convert_csv_to_dataframe(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line_list = eval(line)
            line_str = ' '.join(line_list)
            data.append(line_str)

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


# Get folder path from command line argument
folder_path = sys.argv[1]

# Convert CSV files to DataFrames
df_train = convert_csv_to_dataframe(os.path.join(folder_path, 'train.csv'))
df_val = convert_csv_to_dataframe(os.path.join(folder_path, 'val.csv'))
df_test = convert_csv_to_dataframe(os.path.join(folder_path, 'test.csv'))
df_train_ns = convert_csv_to_dataframe(os.path.join(folder_path, 'train_ns.csv'))
df_val_ns = convert_csv_to_dataframe(os.path.join(folder_path, 'val_ns.csv'))
df_test_ns = convert_csv_to_dataframe(os.path.join(folder_path, 'test_ns.csv'))
train_labels = pd.read_csv(os.path.join(folder_path, 'training_labels.csv'), names=['label'])
val_labels = pd.read_csv(os.path.join(folder_path, 'val_labels.csv'), names=['label'])
test_labels = pd.read_csv(os.path.join(folder_path, 'test_labels.csv'), names=['label'])

# Preprocess the data
tokenizer = CountVectorizer().build_tokenizer()

train_data_embedding = preprocess_data(df_train, tokenizer, word2vec_model)
val_data_embedding = preprocess_data(df_val, tokenizer, word2vec_model)
test_data_embedding = preprocess_data(df_test, tokenizer, word2vec_model)

# Define the neural network architecture
print(train_data_embedding.shape)
input_size = train_data_embedding.shape[1]
output_size = 2

dropout_rates = [0.3, 0.5, 0.7]
l2_lambdas = [0.001, 0.01, 0.1]
activation_functions = ['relu', 'sigmoid', 'tanh']

best_models = {}
best_params = {}
accuracies_table = []

for activation_func in activation_functions:
    best_accuracy = 0.0
    best_model = None

    for dropout_rate in dropout_rates:
        for l2_lambda in l2_lambdas:
            model = keras.Sequential([
                keras.layers.Dense(64, activation=activation_func, input_shape=(input_size,), kernel_regularizer=keras.regularizers.l2(l2_lambda)),
                keras.layers.Dropout(dropout_rate),
                keras.layers.Dense(output_size, activation='softmax')
            ])

            # Compile the model
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Train the model
            batch_size = 128
            num_epochs = 5

            model.fit(train_data_embedding, train_labels, batch_size=batch_size, epochs=num_epochs, validation_data=(val_data_embedding, val_labels))

            # Evaluate the model on the validation set
            val_loss, val_accuracy = model.evaluate(val_data_embedding, val_labels)
            print("Activation Function:", activation_func)
            print("Dropout Rate:", dropout_rate)
            print("Lambda:", l2_lambda)
            print("Validation Accuracy:", val_accuracy)
            print()

            # Check if this model has the best validation accuracy so far
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model
                best_params[activation_func] = {
                    'dropout_rate': dropout_rate,
                    'l2_lambda': l2_lambda
                }

            accuracies_table.append([activation_func, dropout_rate, l2_lambda, val_accuracy])

    # Save the best model for the current activation function
    best_models[activation_func] = best_model
    with open(f"nn_{activation_func}.pkl", 'wb') as file:
        pickle.dump(best_model, file)

# Evaluate the best models on the test set
for activation_func, model in best_models.items():
    test_loss, test_accuracy = model.evaluate(test_data_embedding, test_labels)
    print("Activation Function:", activation_func)
    print("Best Model Parameters:", best_params[activation_func])
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
    print()

# Create a DataFrame for accuracies table
accuracies_df = pd.DataFrame(accuracies_table, columns=['Activation Function', 'Dropout Rate', 'Lambda', 'Validation Accuracy'])
print(accuracies_df)
