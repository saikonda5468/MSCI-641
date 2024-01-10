import sys
import pickle
from sklearn.feature_extraction.text import CountVectorizer

def load_classifier(classifier_type):
    # Define the mapping of classifier types to model file names
    model_files = {
        'mnb_uni': ['mnb_uni.pkl','vectorizer_mnb_uni.pkl'],
        'mnb_bi': ['mnb_bi.pkl','vectorizer_mnb_bi.pkl'],
        'mnb_uni_bi': ['mnb_uni_bi.pkl','vectorizer_mnb_uni_bi.pkl'],
        'mnb_uni_ns': ['mnb_uni_ns.pkl','vectorizer_mnb_uni_bi.pkl'],
        'mnb_bi_ns': ['mnb_bi_ns.pkl','vectorizer_mnb_bi_ns.pkl'],
        'mnb_uni_bi_ns': ['mnb_uni_bi_ns.pkl','vectorizer_mnb_uni_bi_ns.pkl']
    }

    # Load the corresponding model file
    model_file = model_files.get(classifier_type)
    if model_file is None:
        print(f"Invalid classifier type: {classifier_type}")
        sys.exit(1)

    with open(model_file[0], 'rb') as file:
        classifier = pickle.load(file)

    vocabulary_file = model_file[1]
    with open(vocabulary_file, 'rb') as file:
        vectorizer = pickle.load(file)
    return classifier, vectorizer

def classify_sentence(sentence, classifier, vectorizer, remove_stopwords=False):
    # Optionally remove stopwords from the sentence
    if remove_stopwords:
        stop_words = [
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
            "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
            "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was",
            "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the",
            "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
            "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
            "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
        ]

        sentence = ' '.join([word for word in sentence.lower().split() if word.lower() not in stop_words])

    # Vectorize the sentence
    X = vectorizer.transform([sentence])

    # Classify the sentence
    label = classifier.predict(X)[0]

    return label

def main():
    # Read command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python inference.py <path_to_file.txt> <classifier_type>")
        sys.exit(1)
    file_path = sys.argv[1]
    classifier_type = sys.argv[2]

    # Load the classifier
    classifier, vectorizer = load_classifier(classifier_type)

    # Determine if stopwords need to be removed
    remove_stopwords = 'ns' in classifier_type

    # Read sentences from the file
    sentences = []
    with open(file_path, 'r') as file:
        sentences = file.readlines()
    sentences = [sentence.strip() for sentence in sentences]

    # Classify and print the results
    for sentence in sentences:
        label = classify_sentence(sentence.lower(), classifier, vectorizer, remove_stopwords=remove_stopwords)
        print(f"Sentence: {sentence}")
        print(f"Label: {label}")
        print()

if __name__ == "__main__":
    main()
