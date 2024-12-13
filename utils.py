import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def get_dict(data):
    words = sorted(list(set(data)))
    idx = 0
    word2Ind = {}
    Ind2word = {}
    for k in words:
        word2Ind[k] = idx
        Ind2word[idx] = k
        idx += 1
    return word2Ind, Ind2word

def get_idx(words, word2Ind):
    idx = []
    for word in words: idx.append(word2Ind[word])
    return idx

def pack_idx_with_frequency(context_words, word2Ind):
    freq_dict = defaultdict(int)
    for word in context_words: freq_dict[word] += 1

    idxs = get_idx(context_words, word2Ind)

    packed = []
    for i in range(len(idxs)):
        idx = idxs[i]
        freq = freq_dict[context_words[i]]
        packed.append((idx, freq))
    return packed

def get_vectors(data, word2Ind, V, C):
    i = C
    while True:
        y = np.zeros(V)
        x = np.zeros(V)
        center_word = data[i]
        y[word2Ind[center_word]] = 1
        context_words = data[(i - C) : i] + data[(i + 1) : (i + C + 1)]
        num_ctx_words = len(context_words)
        for idx, freq in pack_idx_with_frequency(context_words, word2Ind):
            x[idx] = freq / num_ctx_words
        yield x, y
        i += 1
        if i >= len(data) - C:
            print("i is being set to", C)
            i = C

def get_batches(data, word2Ind, V, C, batch_size):
    batch_x = []
    batch_y = []
    for x, y in get_vectors(data, word2Ind, V, C):
        if len(batch_x) < batch_size:
            batch_x.append(x)
            batch_y.append(y)
        else:
            yield np.array(batch_x).T, np.array(batch_y).T
            batch_x = []
            batch_y = []

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z), axis = 0)

def initialize_model(N, V):
    # X = Vx1 , Y = Vx1, h = Nx1
    W1 = np.random.rand(N, V)
    b1 = np.random.rand(N, 1)

    W2 = np.random.rand(V, N)
    b2 = np.random.rand(V, 1)

    return W1, W2, b1, b2

def forward_prop(x, W1, W2, b1, b2):
    h = np.dot(W1,x) + b1
    h[h < 0] = 0
    z = np.dot(W2,h) + b2
    return z, h

def cross_entropy_loss(y, yhat, m):
    logprobs = np.multiply(np.log(yhat),y)
    cost = - 1/m * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost

def back_prop(X, yhat, y, h, W1, W2, b1, b2, m):

    l1 = np.matmul(W2.T, (yhat - y))
    l1[l1 < 0] = 0 

    grad_W1 = 1/m * np.matmul(l1, X.T)
    grad_W2 = 1/m * np.matmul((yhat - y), h.T)
    grad_b1 = 1/m * np.matmul(np.matmul(W2.T, (yhat - y)), np.ones((m,1)))
    grad_b2 = 1/m * np.matmul((yhat - y), np.ones((m,1)))

    return grad_W1, grad_W2, grad_b1, grad_b2

def gradient_descent(data, word2Ind, N, V, num_iters = 150, m = 128, alpha=0.03):
    W1, W2, b1, b2 = initialize_model(N, V)

    iters = 0
    C = 2

    for x, y in get_batches(data, word2Ind, V, C, m):

        z, h = forward_prop(x, W1, W2, b1, b2)
        yhat = softmax(z)
        cost = cross_entropy_loss(y, yhat, m)

        if ( (iters+1) % 10 == 0):
            print(f"iters: {iters + 1} cost: {cost:.6f}")
            
        grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(x, yhat, y, h, W1, W2, b1, b2, m)
        
        W1 -= alpha*grad_W1
        W2 -= alpha*grad_W2
        b1 -= alpha*grad_b1
        b2 -= alpha*grad_b2

        iters +=1 
        if iters == num_iters: 
            break
        if iters % 100 == 0:
            alpha *= 0.66
            
    return W1, W2, b1, b2

def extract_word_embeddings(W1, W2):
    # W1=(N,V) and W2=(V,N)
    embs = (W1.T + W2)/2.0
    return embs

def compute_pca(data, n_components = 2):
    data = data-data.mean(axis = 0)
    R = np.cov(data, rowvar=False)

    evals, evecs = np.linalg.eigh(R)
    sorted_idx = np.argsort(evals)[::-1][:n_components]

    evecs = evecs[:,sorted_idx]
    evals = evals[sorted_idx]

    return np.dot(evecs.T, data.T).T

def make_pca_plot(embs, word2Ind):

    words = ['king', 'queen','lord','man', 'woman','dog','wolf',
         'rich','happy','sad']

    idx = [word2Ind[word] for word in words]
    X = embs[idx, :]

    result= compute_pca(X, 2)
    
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.show()



    