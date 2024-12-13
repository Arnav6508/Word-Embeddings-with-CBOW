from preprocess import preprocess

def load_data(path):
    with open(path) as file:
        data = file.read()

    data, word2Ind, Ind2word = preprocess(data)
    return data, word2Ind, Ind2word