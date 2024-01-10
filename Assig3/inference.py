import random
import re
import numpy as np
import sys
import gensim.models

file_path = sys.argv[1]
data = []
model = gensim.models.Word2Vec.load("word2vec_model.bin").wv
def read_data():
    with open(file_path, 'r') as file:
        for word in file:
            data.append(word.strip())
    for word in data:
        if word in model.key_to_index:
            similar_words = model.most_similar(word, topn=20)
            print(f"Words similar to '{word}':")
            for similar_word, similarity in similar_words:
                print(f"{similar_word}: {similarity},")
            print()
        else:
            print(f"'{word}' is not in the vocabulary of the Word2Vec model.\n")
            
read_data()
