import string
from nltk.stem import PorterStemmer

def tokenizer(text):
    """Preprocess a string and return a list of clean tokens."""
    stemmer = PorterStemmer()

    translator = str.maketrans('', '', string.punctuation)

    tokens = text.translate(translator).lower().split()
    with open('data/stopwords.txt', 'r') as f:
        stop_words = f.read().splitlines()
    
    for index, token in enumerate(tokens):
        if token in stop_words:
            tokens.remove(token)
            continue
        tokens[index] = stemmer.stem(token)

    return tokens

def has_matching_token(query_tokens, title_tokens):
    """Checks if at least one token from the query matches any part of a token from the title."""
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False