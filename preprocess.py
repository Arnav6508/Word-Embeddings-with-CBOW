import re
import nltk
from nltk.tokenize import word_tokenize

def get_dict(data):
    data = sorted(list(set(data)))
    word2Ind, Ind2Word = {}, {}
    i = 0
    for word in data:
        word2Ind[word] = i
        Ind2Word[i] = word
        i += 1
    return word2Ind, Ind2Word

def preprocess(data):
    data = re.sub(r'[,!?;-]', '.',data)  # Punctuations are replaced by '.' 
    data = nltk.word_tokenize(data)
    data = [ch.lower() for ch in data if ch.isalpha() or ch == '.'] 
    word2Ind, Ind2word = get_dict(data)
    return data, word2Ind, Ind2word
