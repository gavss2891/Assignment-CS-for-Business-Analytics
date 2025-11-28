import pandas as pd
import numpy as np
import re
import os
import csv
from collections import OrderedDict
from sklearn.cluster import AgglomerativeClustering
from ordered_set import OrderedSet
import math
import Levenshtein

def _load_tv_brands():
    """
    Load list of known TV brands from the CSV file.
    Returns a set of lowercase brand names for quick membership checks.
    """
    csv_path = os.path.join(os.path.dirname(__file__), 'television_brands.csv')
    brands = set()

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                brand = row.get('Brand', '').strip()
                if brand:
                    brands.add(brand.lower())
    
    return brands


_TV_BRANDS = sorted(_load_tv_brands(), key=len, reverse=True)

def cosineSim(string_1, string_2):
    string_1_set = set(string_1.split())
    string_2_set = set(string_2.split())

    numerator = len(string_1_set.intersection(string_2_set))

    size_1 = len(string_1_set)
    size_2 = len(string_2_set)

    return numerator/(math.sqrt(size_1) * math.sqrt(size_2)) if size_1 != 0 and size_2 != 0 else 0


def calcSim(s1, s2):
    """
    Calculate the similarity between two strings using q-gram similarity.
    """
    q = 3
    pad_char = "#"

    def generate_qgrams(s):
        # pad with (q-1) chars on both sides
        padded = pad_char * (q - 1) + s.lower() + pad_char * (q - 1)
        return [padded[i:i+q] for i in range(len(padded) - q + 1)]

    qgrams1 = generate_qgrams(s1)
    qgrams2 = generate_qgrams(s2)

    set1, set2 = set(qgrams1), set(qgrams2)

    n1, n2 = len(set1), len(set2)
    intersection = len(set1 & set2)

    qdist = n1 + n2 - 2 * intersection # qGramDistance = |Δ| = n1 + n2 − 2·|intersection|
    denom = n1 + n2

    return (denom - qdist) / denom  if denom != 0 else 0.0 # similarity = (n1 + n2 − qdist) / (n1 + n2) 

def sameShop(p1, p2):
    return p1.get('shop') == p2.get('shop')

def diffBrand(p1, p2):
    """
    Checks 'featuresMap' dictionary for 'Brand' key
    searches the title if Brand is not found.
    """
    p1_brand = "NA"
    p2_brand = "NA"
    
    if p1.get("featuresMap", {}).get("Brand") is not None:
        p1_brand = p1.get("featuresMap").get("Brand").lower()
    
    if p2.get("featuresMap", {}).get("Brand") is not None:
        p2_brand = p2.get("featuresMap").get("Brand").lower()
    
    if p1_brand == "NA":
        title_1 = p1.get("title", "").lower()
        for brand in _TV_BRANDS:
            if re.search(rf'\b{re.escape(brand)}\b', title_1):
                p1_brand = brand
                break
    
    if p2_brand == "NA":
        title_2 = p2.get("title", "").lower()
        for brand in _TV_BRANDS:
            if re.search(rf'\b{re.escape(brand)}\b', title_2):
                p2_brand = brand
                break

    return p1_brand != p2_brand and p1_brand != "NA" and p2_brand != "NA"


def extract_model_words(source, from_title=False):
    """
    Extract model words from either features dict with keys, or from title string.
    """
    if from_title:
        pattern = r'([a-zA-Z0-9]*((\d*\.)?\d+[^0-9, ]+)[a-zA-Z0-9]*)'
        matches = re.findall(pattern, source)
        return [m[0] for m in matches]
    else:
        features, keys = source
        pattern = r'^\d+\.\d+|\b\d+:\d+\b|(?<!\S)\d{3,}\s*[x]\s*\d{3,}(?!\S)'
        model_words = OrderedSet()
        for key in keys:
            if key in features: 
                matches = re.findall(pattern, features.get(key, ""))
                model_words.update(matches)
        return model_words

def parse_model_word(s):
    non_numeric = ''.join(ch for ch in s if not ch.isdigit())
    numeric = ''.join(ch for ch in s if ch.isdigit())
    return non_numeric, numeric

def Levenshtein_sim(s1, s2):
    D = Levenshtein.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 0 if max_len == 0 else 1 - (D / max_len)

def avg_Levenshtein_dist(mw1_data, mw2_data, require_family_match):
    numerator = 0
    denominator = 0

    for (w1, nn1, num1) in mw1_data:
        for (w2, nn2, num2) in mw2_data:

            if not require_family_match or (require_family_match and Levenshtein_sim(nn1, nn2) > 0.5 and num1 == num2):
                numerator += (1 - Levenshtein_sim(w1, w2)) * (len(w1) + len(w2))
                denominator += (len(w1) + len(w2))

    return numerator / denominator if denominator != 0 else 0


def TMWMSim(title1, title2, alpha, beta, delta, epsilon):
    title1 = title1.lower()
    title2 = title2.lower()

    cosine_sim = cosineSim(title1, title2)
    if cosine_sim > alpha:
        return 1

    mw1_list = list(dict.fromkeys(extract_model_words(title1, from_title=True)))
    mw2_list = list(dict.fromkeys(extract_model_words(title2, from_title=True)))

    if len(mw1_list) == 0 and len(mw2_list) == 0:
        return cosine_sim if cosine_sim > beta else -1.0

    mw1_data = [(mw, *parse_model_word(mw)) for mw in mw1_list]
    mw2_data = [(mw, *parse_model_word(mw)) for mw in mw2_list]

    similar_model_words = False

    for (_, nn1, num1) in mw1_data:
        for (_, nn2, num2) in mw2_data:
            approx_sim = Levenshtein_sim(nn1, nn2)

            # Similar family name but conflicting numbers
            if approx_sim > epsilon and num1 != num2:
                return -1
            # Similar family name with matching numbers
            if approx_sim > epsilon and num1 == num2:
                similar_model_words = True

    base_mw_sim = avg_Levenshtein_dist(mw1_data, mw2_data, require_family_match=False)
    final_sim = beta * cosine_sim + (1 - beta) * base_mw_sim

    if similar_model_words:
        mw_only_sim = avg_Levenshtein_dist(mw1_data, mw2_data, require_family_match=True)
        final_sim = delta * mw_only_sim + (1 - delta) * final_sim

    return final_sim


def clustering(dissimilarity_matrix, threshold):
    clustered= AgglomerativeClustering(metric="precomputed", linkage="single", distance_threshold=threshold, n_clusters=None)
    clustered.fit(dissimilarity_matrix)

    return clustered


def main(candidate_pairs, data, gamma, epsilon, mu):
    dissimilarity: pd.DataFrame = candidate_pairs.copy()

    print("Begin MSM...\n")

    for i in range(len(dissimilarity)):
        for j in range(i+1, len(dissimilarity)):       
            row_name = dissimilarity.index[i]
            column_name = dissimilarity.columns[j]

            if dissimilarity.loc[row_name, column_name] == 0:
                product_1 = data.get(row_name)
                product_2 = data.get(column_name)

                if (sameShop(product_1, product_2) or (diffBrand(product_1, product_2))):
                    dissimilarity.loc[row_name, column_name] = 1
                    dissimilarity.loc[column_name, row_name] = 1
                    continue

                sim = mean_sim = m = w = 0  # Initialize similarity metrics

                features_1 = product_1.get("featuresMap")
                features_2 = product_2.get("featuresMap")

                no_match_keys_1 = list(features_1.keys()).copy()
                no_match_keys_2 = list(features_2.keys()).copy()

                for key_1 in list(no_match_keys_1): 
                    for key_2 in list(no_match_keys_2):
                        key_sim = calcSim(s1=key_1, s2=key_2)
                        
                        if key_sim > gamma:
                            value_sim = calcSim(s1=features_1.get(key_1), s2=features_2.get(key_2))
                            sim += key_sim * value_sim
                            m += 1
                            w += key_sim
                            
                            no_match_keys_1.remove(key_1)
                            no_match_keys_2.remove(key_2)
                            break

                if w > 0:
                    mean_sim = sim / w

                model_words_1 = extract_model_words((features_1, no_match_keys_1), from_title=False)
                model_words_2 = extract_model_words((features_2, no_match_keys_2), from_title=False)

                union = model_words_1.union(model_words_2)
                mw_percentage = 0 if len(union) == 0 else len(model_words_1.intersection(model_words_2)) / len(union)

                title_sim = TMWMSim(title1=product_1.get("title"), title2=product_2.get("title"), alpha=0.602, beta=0.0, delta=0.5, epsilon=0.5)

                if title_sim == -1:
                    theta_1 = m/min(len(features_1), len(features_2))
                    theta_2 = 1 - theta_1
                    h_sim = theta_1 * mean_sim + theta_2 * mw_percentage
                else:
                    theta_1 = (1 - mu) * m/min(len(features_1), len(features_2))
                    theta_2 = 1 - mu - theta_1
                    h_sim = theta_1 * mean_sim + theta_2 * mw_percentage + mu * title_sim
                
                dissimilarity.loc[row_name, column_name] = 1 - h_sim
                dissimilarity.loc[column_name, row_name] = 1 - h_sim

    np.fill_diagonal(dissimilarity.values, 0)

    print("Begin clustering...\n")
    return clustering(dissimilarity_matrix=dissimilarity, threshold=epsilon), dissimilarity


if __name__ == "__main__":
    pass