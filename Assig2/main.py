import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import os
import sys

def convert_csv_to_dataframe(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line_list = eval(line)
            line_str = ' '.join(line_list)
            data.append(line_str)

    df = pd.DataFrame({'text': data})
    return df

def extract_features(ngram_range, train_data, val_data, test_data):
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    vectorizer.fit(train_data['text'])
    X_train = vectorizer.transform(train_data['text'])
    X_val = vectorizer.transform(val_data['text'])
    X_test = vectorizer.transform(test_data['text'])
    return X_train, X_val, X_test,vectorizer

def train_and_evaluate_classifier(X_train, X_val, X_test, train_labels, val_labels, test_labels):
    classifier = MultinomialNB()

    # Define the hyperparameter grid
    param_grid = {
        'alpha': [0.1, 0.5, 1.0],  # Example hyperparameters, adjust as needed
        'fit_prior': [True, False]
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(classifier, param_grid, cv=5)

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, train_labels)

    # Get the best classifier with optimal hyperparameters
    best_classifier = grid_search.best_estimator_

    # Use the best classifier to make predictions
    y_pred_val = best_classifier.predict(X_val)
    y_pred_test = best_classifier.predict(X_test)

    # Calculate accuracy scores
    accuracy_val = accuracy_score(val_labels, y_pred_val)
    accuracy_test = accuracy_score(test_labels, y_pred_test)

    return best_classifier, accuracy_val, accuracy_test

# Get the folder path from command-line arguments
if len(sys.argv) < 2:
    print("Please provide the folder path.")
    sys.exit(1)

folder_path = sys.argv[1]

# Convert CSV files to DataFrames
df_out = convert_csv_to_dataframe(os.path.join(folder_path, 'out.csv'))
df_train = convert_csv_to_dataframe(os.path.join(folder_path, 'train.csv'))
df_val = convert_csv_to_dataframe(os.path.join(folder_path, 'val.csv'))
df_test = convert_csv_to_dataframe(os.path.join(folder_path, 'test.csv'))
df_out_ns = convert_csv_to_dataframe(os.path.join(folder_path, 'out_ns.csv'))
df_train_ns = convert_csv_to_dataframe(os.path.join(folder_path, 'train_ns.csv'))
df_val_ns = convert_csv_to_dataframe(os.path.join(folder_path, 'val_ns.csv'))
df_test_ns = convert_csv_to_dataframe(os.path.join(folder_path, 'test_ns.csv'))
train_labels = pd.read_csv(os.path.join(folder_path, 'training_labels.csv'), names=['label'])
val_labels = pd.read_csv(os.path.join(folder_path, 'val_labels.csv'), names=['label'])
test_labels = pd.read_csv(os.path.join(folder_path, 'test_labels.csv'), names=['label'])

# Define the conditions for training and evaluation
conditions = [
    {'stopwords': 'yes', 'ngram_range': (1, 1), 'name': 'with stopwords, unigrams','file_name' : 'mnb_uni' },
    {'stopwords': 'yes', 'ngram_range': (2, 2), 'name': 'with stopwords, bigrams','file_name' : 'mnb_bi'},
    {'stopwords': 'yes', 'ngram_range': (1, 2), 'name': 'with stopwords, unigrams + bigrams','file_name' : 'mnb_uni_bi'},
    {'stopwords': 'No', 'ngram_range': (1, 1), 'name': 'without  Stopwords , unigrams','file_name' : 'mnb_uni_ns'},
    {'stopwords': 'No', 'ngram_range': (2, 2), 'name': 'without Stopwords , bigrams','file_name' : 'mnb_bi_ns'},
    {'stopwords': 'No', 'ngram_range': (1, 2), 'name': 'without Stopwords , unigrams + bigrams','file_name' : 'mnb_uni_bi_ns'}
]

results = []

for condition in conditions:
    stopwords = condition['stopwords']
    ngram_range = condition['ngram_range']
    condition_name = condition['name']
    file_path = condition['file_name']
    
    if stopwords == 'yes':
        # Extract features
        X_train, X_val, X_test,vectorizer = extract_features(ngram_range, df_train, df_val, df_test)

        # Train and evaluate the classifier
        classifier, accuracy_val, accuracy_test = train_and_evaluate_classifier(X_train, X_val, X_test, train_labels['label'], val_labels['label'], test_labels['label'])
        
        model_name = f"data/{file_path}.pkl"
        with open(model_name, 'wb') as file:
            pickle.dump(classifier, file)
        vectorizer_name = f"data/vectorizer_{file_path}.pkl"
        with open(vectorizer_name, 'wb') as file:
            pickle.dump(vectorizer, file)
        # Store the results
        results.append({'Condition': condition_name, 'Accuracy (Val)': accuracy_val, 'Accuracy (Test)': accuracy_test})
    
    else:
        # Extract features
        X_train_ns, X_val_ns, X_test_ns,vectorizer_ns = extract_features(ngram_range, df_train_ns, df_val_ns, df_test_ns)

        # Train and evaluate the classifier
        classifier_ns, accuracy_val_ns, accuracy_test_ns = train_and_evaluate_classifier(X_train_ns, X_val_ns, X_test_ns, train_labels['label'], val_labels['label'], test_labels['label'])
        
        model_name = f"data/{file_path}.pkl"
        with open(model_name, 'wb') as file:
            pickle.dump(classifier_ns, file)
        vectorizer_name = f"data/vectorizer_{file_path}.pkl"
        with open(vectorizer_name, 'wb') as file:
            pickle.dump(vectorizer_ns, file)
        # Store the results
        results.append({'Condition': condition_name, 'Accuracy (Val)': accuracy_val_ns, 'Accuracy (Test)': accuracy_test_ns})
    
# Print the results
results_df = pd.DataFrame(results)
print(results_df)
