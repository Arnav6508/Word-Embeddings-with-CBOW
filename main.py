import pickle
from load import load_data
from utils import gradient_descent, extract_word_embeddings, make_pca_plot

def create_model(train_path, N = 50, num_iters = 150, m = 128):
    '''
        N         : embedding_size
        num_iters : # epcohs
        m         : batch_size
    '''

    data, word2Ind, Ind2word = load_data(train_path)

    V = len(word2Ind)
    W1, W2, b1, b2 = gradient_descent(data, word2Ind, N, V, num_iters, m)
    embs = extract_word_embeddings(W1, W2)

    embeddings = {
        'embs': embs,
        'word2Ind': word2Ind
    }

    with open('embeddings.pkl', 'wb') as file:
        pickle.dump(embeddings, file)

    make_pca_plot(embs, word2Ind)