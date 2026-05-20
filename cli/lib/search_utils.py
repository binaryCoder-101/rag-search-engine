import json

def load_movies():
    json_file = open('data/movies.json') 
    return json.load(json_file)["movies"]

def load_stopwords():
     with open('data/stopwords.txt', 'r') as f:
         return f.read().splitlines()