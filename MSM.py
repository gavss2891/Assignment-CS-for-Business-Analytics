from collections import defaultdict
import re


def calcSim(s1, s2):
    """
    Calculate overlapping Q-Gram Similarity (Jaccard on q-grams) between two strings.
    Uses q=3 for q-gram generation.
    
    Args:
        s1 (str): First string
        s2 (str): Second string
        
    Returns:
        float: Jaccard similarity score between 0.0 and 1.0
    """
    def generate_qgrams(s, q=3):
        """Generate q-grams from a string."""
        if len(s) < q:
            return set([s])
        return set([s[i:i+q] for i in range(len(s) - q + 1)])
    
    qgrams1 = generate_qgrams(s1.lower())
    qgrams2 = generate_qgrams(s2.lower())
    
    if len(qgrams1) == 0 and len(qgrams2) == 0:
        return 1.0
    
    intersection = len(qgrams1 & qgrams2)
    union = len(qgrams1 | qgrams2)
    
    return intersection / union if union > 0 else 0.0


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
    brand1 = p1.get('featuresMap', {}).get('Brand', '')
    brand2 = p2.get('featuresMap', {}).get('Brand', '')
    
    return brand1 != brand2 and brand1 != '' and brand2 != ''


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

