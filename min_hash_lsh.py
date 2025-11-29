import numpy as np 
import pandas as pd
import re
from collections import OrderedDict
from ordered_set import OrderedSet
from primePy import primes


def get_mw(data):
    # regex for model words defined same as in MSMP+ paper
    title_regex = r'([a-zA-Z0-9]*((\d*\.)?\d+[^0-9, ]+)[a-zA-Z0-9]*)'
    kvp_regex = r'^\d+\.\d+|\b\d+:\d+\b|(?<!\S)\d{3,}\s*[x]\s*\d{3,}(?!\S)'

    mw = OrderedSet()

    for key in data:
        product = data.get(key)
        mw.update(x[0] for x in re.findall(title_regex, product.get("title")))

        features = product.get("featuresMap")
        for key in features:
            for m in re.findall(kvp_regex, features.get(key)):
                mw.add(re.sub(r'[^0-9xX.:]', '', m))  # remove non-numeric characters
    return mw


def get_binary_vector(mw, data):
    binary_vectors = OrderedDict()
    
    for key in data:
        product = data.get(key)
        title = product.get("title", "").lower()
        features = product.get("featuresMap", {})
        values = [str(v).lower() for v in features.values()]
        vec = np.zeros(len(mw))
        
        i = 0
        for word in mw:
            if word in title or any(word in val for val in values):
                vec[i] = 1
            else:
                vec[i] = 0
            i += 1
        
        binary_vectors[key] = vec
    
    return binary_vectors


def min_hash(binary_vectors, fraction = 0.5):
    num_vec = len(binary_vectors)
    len_vec = len(binary_vectors.get(list(binary_vectors.keys())[0]))
    
    # need to round n to work better with the bands later
    n = round((fraction * len_vec) / 100) * 100
    signature_m = np.full((n, num_vec), np.inf)
    
    p = 200003
    a = np.random.randint(0, p, size=n)
    b = np.random.randint(0, p, size=n)
    
    vec_array = np.array([binary_vectors[key] for key in binary_vectors])
    for i in range(len_vec):
        hashed_values = (a + b * i) % p
        
        mask = vec_array[:, i] == 1
    
        signature_m[:, mask] = np.minimum(hashed_values[:, None], signature_m[:, mask])
    
    return signature_m


def lsh(signature_m, keys, b):
    n, n_prod = signature_m.shape
    r = n // b
    dissim = np.ones((n_prod, n_prod))
    
    for band_idx in range(b):
        start = band_idx * r
        end = start + r
        band = signature_m[start:end, :]
        
        buckets = {}
        for i in range(n_prod):
            sig = tuple(band[:, i])
            if sig not in buckets:
                buckets[sig] = []
            buckets[sig].append(i)
        
        for items in buckets.values():
            if len(items) > 1:
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        idx_i, idx_j = items[i], items[j]
                        dissim[idx_i, idx_j] = 0.0
                        dissim[idx_j, idx_i] = 0.0

    candidates = pd.DataFrame(dissim, index=keys, columns=keys)

    return candidates


def main(data, fraction=0.5, b = int):
    mw = get_mw(data)
    binary_vectors = get_binary_vector(mw, data)
    signature_m = min_hash(binary_vectors, fraction)
    candidates = lsh(signature_m, data.keys(), b)
    return candidates

if __name__ == "__main__":
    pass

