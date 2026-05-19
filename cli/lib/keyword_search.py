from lib.search_utils import tokenizer, load_movies
from pickle import dump

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, input_text):
        input_tokens = tokenizer(input_text)
        for token in input_tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term):
        doc_ids = list(self.index[term])
        doc_ids.sort()
        return doc_ids
    
    def build(self):
        movies_json = load_movies()

        for movie in movies_json:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self):
        with open("cache/index.pkl", "wb") as file:
            dump(self.index, file)
        with open("cache/docmap.pkl", "wb") as file:
            dump(self.docmap, file)