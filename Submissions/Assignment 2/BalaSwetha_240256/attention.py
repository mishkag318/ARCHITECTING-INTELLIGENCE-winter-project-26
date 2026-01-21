import numpy as np

def attention(Q, K, V):
    dk = K.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(dk)
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    output = np.matmul(weights, V)
    return output, weights


def masked_attention(Q, K, V):
    dk = K.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(dk)

    mask = np.triu(np.ones_like(scores), k=1)
    scores[mask == 1] = -np.inf

    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    output = np.matmul(weights, V)
    return output, weights
