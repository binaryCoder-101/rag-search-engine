import string

def tokenizer(text):
    """Preprocess a string and return a list of clean tokens."""
    
    translator = str.maketrans('', '', string.punctuation)

    return text.translate(translator).lower().split()

def has_matching_token(query_tokens, title_tokens):
    """Checks if at least one token from the query matches any part of a token from the title."""
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False