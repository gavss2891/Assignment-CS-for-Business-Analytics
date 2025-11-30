import pandas as pd
import numpy as np
import re
from sklearn.cluster import AgglomerativeClustering
import math
from TMWMSim import TMWMSim, get_mw

_TV_BRANDS = []

def q_gram_sim(s1, s2):
    q = 3
    q1 = {s1[i:i+q] for i in range(len(s1) - q + 1)}
    q2 = {s2[i:i+q] for i in range(len(s2) - q + 1)}
    intersect = q1.intersection(q2)
    union = q1.union(q2)
    return len(intersect) / len(union) if len(union) != 0 else 0.0 
    
def sameShop(p1, p2):
    return p1.get('shop') == p2.get('shop')

def diffBrand(p1, p2):
    def get_brand(p):
        f = p.get("featuresMap", {})
        brand = (f.get("Brand") or f.get("brand") or "").lower() or "NA"
        if brand == "NA":
            title = p.get("title", "").lower()
            for b in _TV_BRANDS:
                if re.search(rf'\b{re.escape(b)}\b', title):
                    return b
        return brand
    
    b1, b2 = get_brand(p1), get_brand(p2)
    return b1 != b2 and b1 != "NA" and b2 != "NA"

def clustering(dissimilarity_matrix, threshold):
    clustered= AgglomerativeClustering(metric="precomputed", linkage="complete", distance_threshold=threshold, n_clusters=None)
    clustered.fit(dissimilarity_matrix)
    return clustered

def set_tv_brands(tv_brands):
    global _TV_BRANDS
    _TV_BRANDS = [brand.lower() if isinstance(brand, str) else brand for brand in tv_brands]

def main(candidates, data, gamma, epsilon, mu):
    d = candidates.copy()
    print("Running msm ---------\n")

    for i in range(len(d)):
        for j in range(i+1, len(d)):
            r, c = d.index[i], d.columns[j]
            if d.loc[r, c] == 0:
                p1, p2 = data.get(r), data.get(c)
                if sameShop(p1, p2) or diffBrand(p1, p2):
                    d.loc[r, c] = 1
                    d.loc[c, r] = 1
                    continue

                sim = mean_sim = m = w = 0
                f1, f2 = p1.get("featuresMap"), p2.get("featuresMap")
                k1, k2 = list(f1.keys()).copy(), list(f2.keys()).copy()

                for key1 in f1.keys():
                    match = False
                    if not match:
                        for key2 in k2:
                            ks = q_gram_sim(key1, key2)
                            if ks > gamma:
                                vs = q_gram_sim(f1.get(key1), f2.get(key2))
                                sim += ks * vs
                                m += 1
                                w += ks
                                match = True
                                k1.remove(key1)
                                k2.remove(key2)
                                break

                if w > 0:
                    mean_sim = sim / w

                mw1 = get_mw((f1, k1), from_title=False)
                mw2 = get_mw((f2, k2), from_title=False)
                union = mw1.union(mw2)
                mwp = 0 if len(union) == 0 else len(mw1.intersection(mw2)) / len(union)

                ts = TMWMSim(p1.get("title"), p2.get("title"), 0.602, 0.0, 0.5, 0.5)

                if ts == -1:
                    t1 = m / min(len(f1), len(f2))
                    t2 = 1 - t1
                    h = t1 * mean_sim + t2 * mwp
                else:
                    t1 = (1 - mu) * m / min(len(f1), len(f2))
                    t2 = 1 - mu - t1
                    h = t1 * mean_sim + t2 * mwp + mu * ts

                d.loc[r, c] = 1 - h
                d.loc[c, r] = 1 - h

    np.fill_diagonal(d.values, 0)
    print("Start clustering ---------\n")
    return clustering(d, epsilon), d

if __name__ == "__main__":
    pass