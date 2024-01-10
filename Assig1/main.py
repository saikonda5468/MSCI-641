import random
import re
import numpy as np
import sys

class TextDataProcessor:
    def __init__(self, negative_file_path, positive_file_path):
        self.negative_file_path = negative_file_path
        self.positive_file_path = positive_file_path
        self.negative_data = []
        self.positive_data = []
        self.negative_data_cleaned = []
        self.positive_data_cleaned = []
        self.combined_data = []
        self.tokenized_data = []
        self.with_stopwords = []
        self.without_stopwords = []
        self.train_data_without = []
        self.val_data_without = []
        self.test_data_without = []
        self.train_data_with = []
        self.val_data_with = []
        self.test_data_with = []
        self.training_labels = []
        self.val_labels = []
        self.test_labels = []
        self.labels = []

    def read_data(self):
        # Read negative data
        with open(self.negative_file_path, 'r') as file:
            for line in file:
                self.negative_data.append(line.strip())

        # Read positive data
        with open(self.positive_file_path, 'r') as file:
            for line in file:
                self.positive_data.append(line.strip())

    def remove_special_characters(self, corpus):
        special_chars = r'!"\'#$%&()*+/:;,<=>@[\\]^`{|}~'
        cleaned_corpus = []
        for text in corpus:
            cleaned_doc = re.sub('[' + re.escape(special_chars) + ']', '', text.lower())
            cleaned_corpus.append([cleaned_doc])
        return cleaned_corpus
    def split_data(self, processed_data):
        tokenized_data = []
        for sentence_list in processed_data:
            tokenized_sentence = []
            for sentence in sentence_list:
                words = sentence.split()
                tokenized_sentence.append(words)
            tokenized_data.append(tokenized_sentence)
        return tokenized_data

    def create_datasets(self, corpus):
        stop_words = stopwords_list = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
    'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
]
        datasets = {
            'with_stopwords': [],
            'without_stopwords': []
        }
        for sentence_list in corpus:
            sentence = sentence_list[0]  # Get the sentence from the single-element list
            with_stopwords = sentence  # Store the sentence as is
            without_stopwords = [token for token in sentence if token.lower() not in stop_words]
            datasets['with_stopwords'].append(with_stopwords)
            datasets['without_stopwords'].append(without_stopwords)
        return datasets

    def split_data(self, processed_data):
        tokenized_data = []
        for sentence_list in processed_data:
            tokenized_sentence = []
            for sentence in sentence_list:
                words = sentence.split()
                tokenized_sentence.append(words)
            tokenized_data.append(tokenized_sentence)
        return tokenized_data

    def training_data(self, data, train_ratio, val_ratio, test_ratio):

        # Calculate the number of examples for each set
        num_examples = len(data)
        num_train = int(train_ratio * num_examples)
        num_val = int(val_ratio * num_examples)
        num_test = int(test_ratio * num_examples)

        # Split the data into train, validation, and test sets
        train_data = data[:num_train]
        val_data = data[num_train:num_train + num_val]
        test_data = data[num_train + num_val:]

        return train_data, val_data, test_data

    def labels_data(self, labelling, train_ratio, val_ratio, test_ratio):
        num_examples = len(labelling)
        num_train = int(train_ratio * num_examples)
        num_val = int(val_ratio * num_examples)
        num_test = int(test_ratio * num_examples)

        label_train_data = labelling[:num_train]
        label_val_data = labelling[num_train:num_train + num_val]
        label_test_data = labelling[num_train + num_val:]


        return label_train_data, label_val_data, label_test_data

    def process_data(self):
        self.read_data()

        # Remove special characters from negative data
        self.negative_data_cleaned = self.remove_special_characters(self.negative_data)

        # Remove special characters from positive data
        self.positive_data_cleaned = self.remove_special_characters(self.positive_data)

        # Assign labels to the data
        self.negative_data_cleaned = [(text, 0) for text in self.negative_data_cleaned]
        self.positive_data_cleaned = [(text, 1) for text in self.positive_data_cleaned]

        # Combine and shuffle the data
        self.combined_data = self.negative_data_cleaned + self.positive_data_cleaned
        self.combined_data_before, self.labels_before = zip(*self.combined_data)
        random.shuffle(self.combined_data)
        

        # Separate the shuffled data and labels
        self.combined_data, self.labels = zip(*self.combined_data)
        self.tokenized_data = self.split_data(self.combined_data)

        datasets = self.create_datasets(self.tokenized_data)
        self.with_stopwords = datasets['with_stopwords']
        self.without_stopwords = datasets['without_stopwords']

    def split_datasets(self, train_ratio, val_ratio, test_ratio):
        self.train_data_without, self.val_data_without, self.test_data_without = self.training_data(
            self.without_stopwords, train_ratio, val_ratio, test_ratio
        )
        self.train_data_with, self.val_data_with, self.test_data_with = self.training_data(
            self.with_stopwords, train_ratio, val_ratio, test_ratio
        )
        self.training_labels,self.val_labels,self.test_labels = self.labels_data(self.labels,train_ratio, val_ratio, test_ratio)


        
    def save_datasets_to_excel(self):
        # Convert the datasets to NumPy arrays
        train_data_without = np.array(self.train_data_without, dtype='object')
        val_data_without = np.array(self.val_data_without, dtype='object')
        test_data_without = np.array(self.test_data_without, dtype='object')
        train_data_with = np.array(self.train_data_with, dtype='object')
        val_data_with = np.array(self.val_data_with, dtype='object')
        test_data_with = np.array(self.test_data_with, dtype='object')
        label_training = np.array(self.training_labels, dtype='object')
        label_val = np.array(self.val_labels, dtype='object')
        labe_test = np.array(self.test_labels, dtype='object')
        original_labels = np.array(self.labels_before, dtype='object')
        tokens_with_stop = np.array(self.with_stopwords, dtype='object')
        tokens_without_stop = np.array(self.without_stopwords, dtype='object')
        

        # Define the file paths for the Excel sheets
        folder_path = 'data/'

        train_file_path = folder_path + 'train.csv'
        val_file_path = folder_path + 'val.csv'
        test_file_path = folder_path + 'test.csv'
        train_file_path_without = folder_path + 'train_ns.csv'
        val_file_path_without = folder_path + 'val_ns.csv'
        test_file_path_without = folder_path + 'test_ns.csv'
        training_labels_path = folder_path + 'training_labels.csv'
        test_labels_path = folder_path + 'test_labels.csv'
        val_labels_path = folder_path + 'val_labels.csv'
        labels_path = folder_path + 'labels_out.csv'
        token_with = folder_path + 'out.csv'

        token_without = folder_path + 'out_ns.csv'

        # Save the datasets to Excel
        np.savetxt(train_file_path, train_data_with, fmt='%s')
        np.savetxt(val_file_path, val_data_with, fmt='%s')
        np.savetxt(test_file_path, test_data_with, fmt='%s')
        np.savetxt(train_file_path_without, train_data_without, fmt='%s')
        np.savetxt(val_file_path_without, val_data_without, fmt='%s')
        np.savetxt(test_file_path_without, test_data_without, fmt='%s')
        np.savetxt(training_labels_path, label_training, fmt='%s')
        np.savetxt(val_labels_path, label_val, fmt='%s')
        np.savetxt(test_labels_path, labe_test, fmt='%s')
        np.savetxt(token_with, tokens_with_stop, fmt='%s')
        np.savetxt(token_without, tokens_without_stop, fmt='%s')
        np.savetxt(labels_path, original_labels, fmt='%s')
        
        print("Datasets saved to Excel successfully.")


# Usage example
negative_file_path = sys.argv[1]  # Provide the negative data file path as the first command line argument
positive_file_path = sys.argv[2]  # Provide the positive data file path as the second command line argument

text_processor = TextDataProcessor(negative_file_path, positive_file_path)
text_processor.process_data()
text_processor.split_datasets(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
text_processor.save_datasets_to_excel()
