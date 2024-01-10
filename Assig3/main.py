import random
import re
import numpy as np
import sys
import gensim.models


class word2vec():
    def __init__(self,neg_path,pos_path):
        self.neg_path = neg_path
        self.pos_path = pos_path
        self.corpus = []
    def read_data(self):
        with open(self.neg_path, 'r') as file:
            for line in file:
                self.corpus.append(line.strip())

        with open(self.pos_path, 'r') as file:
            for line in file:
                self.corpus.append(line.strip())
        return self.corpus

    def remove_special_characters(self,corpus):
        special_chars = r'!"\'#$%&()*+/:;,<=>@[\\]^`{|}~'
        cleaned_corpus = []
        for text in corpus:
            cleaned_doc = re.sub('[' + re.escape(special_chars) + ']', '', text.lower())
            cleaned_corpus.append(cleaned_doc)
        return cleaned_corpus

    def split_data(self,processed_data):
        tokenized_data = []
        for sentence in processed_data:
            words = sentence.split()
            tokenized_data.append(words)
        return tokenized_data
    
    def training(self):
        corpus = self.read_data()
        remone = self.remove_special_characters(corpus)
        print(remone[0:10]) 
        tokens = self.split_data(remone)
        print(tokens[0:10])
        model = gensim.models.Word2Vec(tokens, vector_size=100, window=5, min_count=5, workers=4)
        model.save("word2vec_model.bin")
        # Get the most similar words to 'good'
        similar_words_good = model.wv.most_similar('good', topn=20)

        # Get the most similar words to 'bad'
        similar_words_bad = model.wv.most_similar('bad', topn=20)

        print("Words similar to 'good':")
        for word, similarity in similar_words_good:
            print(f"- {word}: {similarity}")

        print("\nWords similar to 'bad':")
        for word, similarity in similar_words_bad:
            print(f"- {word}: {similarity}")
            
            
if __name__ == '__main__':
    w2v = word2vec("C:/Users/saiko/OneDrive/Desktop/641/Assig3/neg.txt", "C:/Users/saiko/OneDrive/Desktop/641/Assig3/pos.txt")
    w2v.training()

    