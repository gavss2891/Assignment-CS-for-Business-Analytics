from collections import defaultdict
import csv
import os
import re


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


_TV_BRANDS = _load_tv_brands()
_TV_BRANDS_SORTED = sorted(_TV_BRANDS, key=len, reverse=True)
print(_TV_BRANDS_SORTED)

def calcSim(s1, s2):
    """
    Overlapping q-gram similarity. 
    Uses q=3 and pads with q-1 dummy characters at both ends.
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

    # qGramDistance = |Δ| = n1 + n2 − 2·|intersection|
    qdist = n1 + n2 - 2 * intersection

    # similarity = (n1 + n2 − qdist) / (n1 + n2) = 2·intersection / (n1 + n2)
    denom = n1 + n2
    if denom == 0:
        return 1.0

    return (denom - qdist) / denom


def sameShop(p1, p2):
    """
    Check if two product structures have the same shop name.
    
    Args:
        p1 (dict): First product structure with 'shop' key
        p2 (dict): Second product structure with 'shop' key
        
    Returns:
        bool: True if shops match, False otherwise
    """
    return p1.get('shop') == p2.get('shop')

def _normalize_brand(brand_value):
    """
    Normalize a brand string by matching it against the known brand list.
    Returns the matched brand in lowercase, or the cleaned brand string if unknown.
    """
    if not brand_value:
        return ''

    cleaned = brand_value.strip().lower()
    if not cleaned:
        return ''

    if cleaned in _TV_BRANDS:
        return cleaned

    # Try partial/substring matches for cases like "Samsung Electronics"
    for known_brand in _TV_BRANDS:
        if known_brand in cleaned or cleaned in known_brand:
            return known_brand

    return cleaned


def _collect_product_strings(product):
    """
    Traverse the product structure and collect all string values (including keys).
    """
    collected = []

    def _walk(value):
        if isinstance(value, str):
            collected.append(value)
        elif isinstance(value, dict):
            for k, v in value.items():
                _walk(k)
                _walk(v)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                _walk(item)

    _walk(product)
    return collected


def _detect_brand(product):
    """
    Inspect every string contained in the product structure and return the first
    known brand that appears as a substring (case-insensitive).
    """
    strings = _collect_product_strings(product)
    for text in strings:
        if not text:
            continue
        lowered = text.lower()
        for brand in _TV_BRANDS_SORTED:
            if brand in lowered:
                return brand
    return ''


def diffBrand(p1, p2):
    """
    Check if two product structures have different brand names.
    Checks 'featuresMap' dictionary for 'Brand' key.
      
    Args:
        p1 (dict): First product structure with 'featuresMap' key
        p2 (dict): Second product structure with 'featuresMap' key
        
    Returns:
        bool: True if brands are different, False if same or either missing
    """    
    # Attempt to match an explicit Brand value first
    brand1_raw = p1.get('featuresMap', {}).get('Brand', '')
    brand2_raw = p2.get('featuresMap', {}).get('Brand', '')

    norm_brand1 = _normalize_brand(brand1_raw) or _detect_brand(p1)
    norm_brand2 = _normalize_brand(brand2_raw) or _detect_brand(p2)

    return norm_brand1 != norm_brand2 and norm_brand1 != '' and norm_brand2 != ''


def key(kvp):
    """
    Extract the key string from a KVP (Key-Value Pair) structure.
    
    Args:
        kvp: Either a tuple (key, value) or dictionary with 'key' entry
        
    Returns:
        str: The key string
    """
    if isinstance(kvp, tuple):
        return kvp[0]
    elif isinstance(kvp, dict):
        return kvp.get('key', '')
    return ''


def value(kvp):
    """
    Extract the value string from a KVP (Key-Value Pair) structure.
    
    Args:
        kvp: Either a tuple (key, value) or dictionary with 'value' entry
        
    Returns:
        str: The value string
    """
    if isinstance(kvp, tuple):
        return kvp[1]
    elif isinstance(kvp, dict):
        return kvp.get('value', '')
    return ''


def exMW(product):
    """
    Extract all model words from the attribute values of a product structure.
    Model words are tokens containing both letters and numbers (e.g., '120Hz', '4K', '55inch').
    
    Args:
        product (dict): Product structure with 'featuresMap' key (dictionary of features)
        
    Returns:
        set: Set of model words (strings) found in all attribute values
    """
    model_words = set()
    features_map = product.get('featuresMap', {})
    
    # Pattern to match tokens with both letters and numbers
    pattern = re.compile(r'\b\w*[a-zA-Z]\w*\d\w*|\w*\d\w*[a-zA-Z]\w*\b')
    
    # Iterate over featuresMap dictionary
    for key, val in features_map.items():
        if val:
            val_str = str(val)
            # Tokenize the value string and find model words
            tokens = re.findall(r'\b\w+\b', val_str)
            for token in tokens:
                if pattern.match(token):
                    model_words.add(token)
    
    return model_words


def mw(set_C, set_D):
    """
    Calculate the percentage of matching model words between two sets (Jaccard Similarity).
    
    Args:
        set_C (set): First set of model words
        set_D (set): Second set of model words
        
    Returns:
        float: Jaccard similarity score between 0.0 and 1.0
    """
    if len(set_C) == 0 and len(set_D) == 0:
        return 1.0
    
    intersection = len(set_C & set_D)
    union = len(set_C | set_D)
    
    return intersection / union if union > 0 else 0.0


# This needs checking!!!
def TMWMSim(p1, p2, alpha, beta):
    """
    Title Model Words Method (TMWM) Similarity.
    Returns 1.0 if title cosine similarity >= alpha,
    a calculated model word Jaccard if model word Jaccard >= beta,
    or -1.0 otherwise.
    
    Uses Jaccard on tokens/model words as proxy for cosine/model word comparison.
    
    Args:
        p1 (dict): First product structure with 'title' key
        p2 (dict): Second product structure with 'title' key
        alpha (float): Threshold for title similarity
        beta (float): Threshold for model word Jaccard similarity
        
    Returns:
        float: Similarity score (1.0, calculated Jaccard, or -1.0)
    """
    # Extract title from products
    title1 = p1.get('title', '')
    title2 = p2.get('title', '')
    
    # Calculate title similarity using Jaccard on tokens (proxy for cosine)
    title_tokens1 = set(re.findall(r'\b\w+\b', title1.lower()))
    title_tokens2 = set(re.findall(r'\b\w+\b', title2.lower()))
    
    if len(title_tokens1) == 0 and len(title_tokens2) == 0:
        title_sim = 1.0
    else:
        intersection = len(title_tokens1 & title_tokens2)
        union = len(title_tokens1 | title_tokens2)
        title_sim = intersection / union if union > 0 else 0.0
    
    # Check if title similarity >= alpha
    if title_sim >= alpha:
        return 1.0
    
    # Calculate model word Jaccard similarity
    mw1 = exMW(p1)
    mw2 = exMW(p2)
    mw_sim = mw(mw1, mw2)
    
    # Check if model word Jaccard >= beta
    if mw_sim >= beta:
        return mw_sim
    
    # Otherwise return -1.0
    return -1.0

# Need to check this!!!
def minFeatures(p1, p2):
    """
    Return the minimum number of Key-Value Pairs (KVPs) between two product structures.
    
    Args:
        p1 (dict): First product structure with 'featuresMap' key
        p2 (dict): Second product structure with 'featuresMap' key
        
    Returns:
        int: Minimum number of features between the two products
    """
    features1 = p1.get('featuresMap', {})
    features2 = p2.get('featuresMap', {})
    return min(len(features1), len(features2))


def hClustering(dist, epsilon):
    """
    Agglomerative Single-Linkage Hierarchical Clustering.
    Stops merging when minimum inter-cluster distance > epsilon.
    
    Args:
        dist (list[list[float]]): The pre-computed dissimilarity matrix (1 - hSim).
        epsilon (float): Threshold distance - clustering stops when min inter-cluster distance > epsilon
        
    Returns:
        list: List of final clusters, where each cluster is a set of indices
    """
    n = len(dist)
    if n == 0:
        return []
    
    # Initialize: each item is its own cluster
    clusters = [{i} for i in range(n)]
    
    # Helper function to get distance between two clusters (single-linkage)
    def cluster_distance(c1, c2):
        """Calculate minimum distance between any pair of items from two clusters (single-linkage)."""
        min_dist = float('inf')
        # Iterate over all pairs of products, one from each cluster
        for i in c1:
            for j in c2:
                # dist[i][j] is the dissimilarity score (1 - hSim)
                min_dist = min(min_dist, dist[i][j])
        return min_dist
    
    # Agglomerative clustering
    while True:
        # Find the two closest clusters
        min_dist = float('inf')
        merge_i, merge_j = -1, -1
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = cluster_distance(clusters[i], clusters[j])
                if d < min_dist:
                    min_dist = d
                    merge_i, merge_j = i, j
        
        # Stop if minimum inter-cluster distance > epsilon or no clusters to merge
        if merge_i == -1 or min_dist > epsilon:
            break
        
        # Merge the two closest clusters
        # Since merge_i < merge_j due to the loop structure, we can safely merge and then pop.
        clusters[merge_i] = clusters[merge_i] | clusters[merge_j]
        clusters.pop(merge_j)
        
        # If only one cluster remains, we're done
        if len(clusters) <= 1:
            break
    
    return clusters


def MSM_algorithm(S, alpha=0.3, beta=0.3, gamma=0.75, epsilon=0.5, mu=0.6):
    """
    Multi-component Similarity Method (MSM) Algorithm.
    
    Computes a multi-component similarity between products from different shops
    and performs hierarchical clustering to group duplicate products.
    
    Args:
        S (dict): Dictionary where keys are shop names and values are lists of products.
                  Each product is a dict with 'shop', 'title', 'featuresMap', etc.
        alpha (float): Threshold for TMWMSim title similarity
        beta (float): Threshold for TMWMSim model word Jaccard similarity
        gamma (float): Threshold similarity for two keys to be considered equal
        epsilon (float): Dissimilarity threshold for hierarchical clustering
        mu (float): Fixed weight of the TMWMSim similarity when it returns a value
        
    Returns:
        list: List of clusters, where each cluster is a set of product indices
    """
    # Step 1: Create a flat list of all products and track which shop each belongs to
    all_products = []
    product_to_shop = {}  # Maps product index to shop name
    shop_to_indices = {}  # Maps shop name to list of product indices
    shop_product_to_global_idx = {}  # Maps (shop_name, local_idx) to global index
    
    for shop_name, products in S.items():
        shop_indices = []
        for local_idx, product in enumerate(products):
            global_idx = len(all_products)
            all_products.append(product)
            product_to_shop[global_idx] = shop_name
            shop_indices.append(global_idx)
            shop_product_to_global_idx[(shop_name, local_idx)] = global_idx
        shop_to_indices[shop_name] = shop_indices
    
    n = len(all_products)
    
    # Step 2: Initialize dissimilarity matrix
    # Set diagonal to 0 (distance from product to itself)
    dist = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0.0
    
    # Step 3: Compute dissimilarity for each pair of products from different shops
    for k, shop_name in enumerate(S.keys()):
        shop_k_products = S[shop_name]
        
        for i, pi in enumerate(shop_k_products):
            # Get the global index of this product
            pi_idx = shop_product_to_global_idx.get((shop_name, i))
            if pi_idx is None:
                continue
            
            # Compare with all products from other shops (S \ Sk)
            for j, pj in enumerate(all_products):
                pj_shop = product_to_shop[j]
                
                # Skip if same shop or same product
                if pj_shop == shop_name or pi_idx == j:
                    continue
                
                # Skip if already computed (to avoid duplicate computation)
                if dist[pi_idx][j] != float('inf'):
                    continue
                
                # Check if same shop or different brand
                if sameShop(pi, pj) or diffBrand(pi, pj):
                    dist[pi_idx][j] = float('inf')
                    dist[j][pi_idx] = float('inf')  # Make symmetric
                else:
                    # Calculate multi-component similarity
                    # Initialize variables
                    sim = 0.0
                    avgSim = 0.0
                    m = 0  # number of matches
                    w = 0.0  # weight of matches
                    
                    # Initialize non-matching keys as all KVPs from both products
                    features_i = pi.get('featuresMap', {})
                    features_j = pj.get('featuresMap', {})
                    
                    # Convert featuresMap to list of KVPs (as tuples)
                    nmki = [(k, v) for k, v in features_i.items()]
                    nmkj = [(k, v) for k, v in features_j.items()]
                    
                    # Match KVPs based on key similarity
                    matched_i = set()
                    matched_j = set()
                    
                    for q_idx, q in enumerate(nmki):
                        if q_idx in matched_i:
                            continue
                        key_q = key(q)
                        
                        for r_idx, r in enumerate(nmkj):
                            if r_idx in matched_j:
                                continue
                            key_r = key(r)
                            
                            # Calculate key similarity
                            keySim = calcSim(key_q, key_r)
                            
                            if keySim > gamma:
                                # Keys are similar enough, match them
                                valueSim = calcSim(value(q), value(r))
                                weight = keySim
                                
                                sim += weight * valueSim
                                m += 1
                                w += weight
                                
                                # Mark as matched
                                matched_i.add(q_idx)
                                matched_j.add(r_idx)
                                break  # Each KVP from pi matches at most one from pj
                    
                    # Calculate average similarity
                    if w > 0:
                        avgSim = sim / w
                    
                    # Get remaining non-matching keys
                    remaining_i = [nmki[i] for i in range(len(nmki)) if i not in matched_i]
                    remaining_j = [nmkj[i] for i in range(len(nmkj)) if i not in matched_j]
                    
                    # Extract model words from remaining non-matching keys
                    # Create temporary products with only remaining features
                    temp_pi = {'featuresMap': {k: v for k, v in remaining_i}}
                    temp_pj = {'featuresMap': {k: v for k, v in remaining_j}}
                    
                    mw_i = exMW(temp_pi)
                    mw_j = exMW(temp_pj)
                    mwPerc = mw(mw_i, mw_j)
                    
                    # Calculate title similarity
                    titleSim = TMWMSim(pi, pj, alpha, beta)
                    
                    # Calculate hSim based on whether titleSim is -1
                    if titleSim == -1.0:
                        # Title similarity doesn't meet threshold
                        theta1 = m / minFeatures(pi, pj) if minFeatures(pi, pj) > 0 else 0.0
                        theta2 = 1.0 - theta1
                        hSim = theta1 * avgSim + theta2 * mwPerc
                    else:
                        # Title similarity meets threshold
                        min_feat = minFeatures(pi, pj)
                        if min_feat > 0:
                            theta1 = (1.0 - mu) * m / min_feat
                        else:
                            theta1 = 0.0
                        theta2 = 1.0 - mu - theta1
                        hSim = theta1 * avgSim + theta2 * mwPerc + mu * titleSim
                    
                    # Convert similarity to dissimilarity
                    dist[pi_idx][j] = 1.0 - hSim
                    dist[j][pi_idx] = 1.0 - hSim  # Make symmetric
    
    # Step 4: Perform hierarchical clustering
    clusters = hClustering(dist, epsilon)
    
    return clusters

