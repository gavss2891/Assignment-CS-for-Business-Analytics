"""
Test script for MSM algorithm on subset_data.json
"""
import json
from MSM import MSM_algorithm, diffBrand

def load_and_organize_data(filename='subset_data.json'):
    """
    Load data from JSON file and organize by shop.
    
    The JSON structure is:
    {
        "modelID": [product1, product2, ...],
        ...
    }
    
    We need to reorganize it as:
    {
        "shop_name": [product1, product2, ...],
        ...
    }
    """
    print("Loading data from", filename, "...")
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Reorganize by shop
    S = {}  # Dictionary: shop_name -> list of products
    
    for model_id, products in data.items():
        for product in products:
            shop_name = product.get('shop')
            if shop_name:
                if shop_name not in S:
                    S[shop_name] = []
                S[shop_name].append(product)
    
    print(f"Loaded {len(data)} model IDs")
    print(f"Organized into {len(S)} shops")
    
    # Print shop statistics
    total_products = 0
    for shop_name, products in S.items():
        total_products += len(products)
        print(f"  {shop_name}: {len(products)} products")
    
    print(f"Total products: {total_products}")
    
    return S

def main():
    """Main function to run MSM algorithm on subset data"""
    # Load and organize data
    S = load_and_organize_data('subset_data.json')
    
    # Set algorithm parameters
    alpha = 0.6   # Threshold for TMWM title similarity
    beta = 0    # Threshold for TMWM model word Jaccard similarity
    gamma = 0.75  # Threshold similarity for two keys to be considered equal
    epsilon = 0.5 # Dissimilarity threshold for hierarchical clustering
    mu = 0.65      # Fixed weight of the TMWM similarity
    
    print("\n" + "="*80)
    print("MSM Algorithm Parameters:")
    print("="*80)
    print(f"  alpha (TMWM title threshold): {alpha}")
    print(f"  beta (TMWM model word threshold): {beta}")
    print(f"  gamma (key similarity threshold): {gamma}")
    print(f"  epsilon (clustering dissimilarity threshold): {epsilon}")
    print(f"  mu (TMWM weight): {mu}")
    print("="*80 + "\n")
    
    # Run MSM algorithm
    print("Running MSM algorithm...")
    clusters = MSM_algorithm(S, alpha=alpha, beta=beta, gamma=gamma, 
                            epsilon=epsilon, mu=mu)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total clusters found: {len(clusters)}")
    
    # Show cluster size distribution
    cluster_sizes = [len(c) for c in clusters]
    if cluster_sizes:
        print(f"\nCluster size statistics:")
        print(f"  Min cluster size: {min(cluster_sizes)}")
        print(f"  Max cluster size: {max(cluster_sizes)}")
        print(f"  Average cluster size: {sum(cluster_sizes) / len(cluster_sizes):.2f}")
        
        # Count clusters by size
        size_distribution = {}
        for size in cluster_sizes:
            size_distribution[size] = size_distribution.get(size, 0) + 1
        
        print(f"\nCluster size distribution:")
        for size in sorted(size_distribution.keys()):
            print(f"  Size {size}: {size_distribution[size]} clusters")
    
    # Show details of clusters (first 10)
    print("\n" + "="*80)
    print("CLUSTER DETAILS (first 10 clusters)")
    print("="*80)
    
    # Create a flat list of all products for indexing
    all_products = []
    for shop_name, products in S.items():
        all_products.extend(products)
    
    for i, cluster in enumerate(clusters[:10]):
        print(f"\nCluster {i+1} (size: {len(cluster)}):")
        for idx in sorted(cluster):
            product = all_products[idx]
            print(f"  [{idx}] {product.get('shop')} - {product.get('modelID')} - {product.get('title')[:60]}...")
    
    if len(clusters) > 10:
        print(f"\n... ({len(clusters) - 10} more clusters)")
    
    # Verify clustering: products with same modelID should ideally be in same cluster
    print("\n" + "="*80)
    print("VERIFICATION: Model ID Distribution in Clusters")
    print("="*80)
    
    # Map model IDs to clusters
    model_to_clusters = {}
    for cluster_idx, cluster in enumerate(clusters):
        for product_idx in cluster:
            product = all_products[product_idx]
            model_id = product.get('modelID')
            if model_id not in model_to_clusters:
                model_to_clusters[model_id] = []
            model_to_clusters[model_id].append(cluster_idx)
    
    # Check how many model IDs are split across clusters
    split_models = []
    for model_id, cluster_indices in model_to_clusters.items():
        unique_clusters = set(cluster_indices)
        if len(unique_clusters) > 1:
            split_models.append((model_id, unique_clusters))
    
    print(f"Total unique model IDs: {len(model_to_clusters)}")
    print(f"Model IDs split across multiple clusters: {len(split_models)}")
    if split_models:
        print("\nSplit model IDs:")
        for model_id, clusters_set in split_models[:5]:
            print(f"  {model_id}: appears in clusters {sorted(clusters_set)}")

    print("\n" + "="*80)
    print("diffBrand sanity checks")
    print("="*80)

    return clusters


if __name__ == "__main__":
    clusters = main()

