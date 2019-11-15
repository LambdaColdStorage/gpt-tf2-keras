import numpy as np

def find_top_p(inputs, p, temperature, min_k = 1):
    epsilon = 1e-4
    indices, probs = np.array(list(map(lambda x: x[1], inputs))), np.array(list(map(lambda x: x[0], inputs)))

    # Softmax
    probs = probs / temperature
    probs = probs - np.max(probs)
    probs = np.exp(probs)
    probs = probs / np.sum(probs)

    # Cut off by accumulated probabilities
    x = np.cumsum(probs)
    m = np.where(x < p)

    if m[0].size == 0:
        m = (np.array(range(min_k)),)

    indices = indices[m[0]]
    probs = probs[m[0]]

    # Sample
    # https://github.com/numpy/numpy/issues/8317
    probs /= (1.0 + epsilon) * probs.sum()
    s = np.random.multinomial(1, probs)
    
    return indices[np.where(s == 1)][0]


def find_top_k(inputs, k, temperature):
    epsilon = 1e-4
    indices, probs = np.array(list(map(lambda x: x[1], inputs))), np.array(list(map(lambda x: x[0], inputs)))

    # Softmax
    probs = probs / temperature
    probs = probs - np.max(probs)
    probs = np.exp(probs)
    probs = probs / np.sum(probs)

    indices = indices[:k]
    probs = probs[:k]

    # Sample
    # https://github.com/numpy/numpy/issues/8317
    probs /= (1.0 + epsilon) * probs.sum()
    s = np.random.multinomial(1, probs)
    
    return indices[np.where(s == 1)][0]