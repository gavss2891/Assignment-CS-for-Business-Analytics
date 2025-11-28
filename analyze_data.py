import json
from collections import Counter

f = open('TVs-all-merged.json', 'r', encoding='utf-8')
data = json.load(f)
f.close()

# Collect shops and feature names
shops = []
feature_name_counts = Counter()
products_with_featuresMap = 0
total_products = 0

# Iterate through all model IDs and their products
for model_id, products in data.items():
    for product in products:
        total_products += 1
        # Extract shop
        if 'shop' in product:
            shops.append(product['shop'])
        
        # Extract feature names from featuresMap
        if 'featuresMap' in product and isinstance(product['featuresMap'], dict):
            products_with_featuresMap += 1
            for feature_name in product['featuresMap'].keys():
                feature_name_counts[feature_name] += 1

# Count unique shops
shop_counts = Counter(shops)

# Print results
print("=" * 60)
print("UNIQUE SHOPS AND THEIR COUNTS")
print("=" * 60)
for shop, count in sorted(shop_counts.items()):
    print(f"{shop}: {count}")

print(f"\nTotal unique shops: {len(shop_counts)}")
print(f"Total product entries: {sum(shop_counts.values())}")

print("\n" + "=" * 60)
print(f"Products with featuresMap: {products_with_featuresMap} out of {total_products} total products")
print("=" * 60)

print("\n" + "=" * 60)
print("UNIQUE FEATURE NAMES IN featuresMap (Ranked by Count)")
print("=" * 60)
# Sort by count descending, then alphabetically
sorted_features = sorted(feature_name_counts.items(), key=lambda x: (-x[1], x[0]))

for feature_name, count in sorted_features:
    print(f"{feature_name}: {count} products")

print(f"\nTotal unique feature names: {len(feature_name_counts)}")

# Find duplicates sold in the same store
print("\n" + "=" * 60)
print("DUPLICATES SOLD IN THE SAME STORE")
print("=" * 60)

duplicates_in_same_store = []

for model_id, products in data.items():
    # Group products by shop
    shop_products = {}
    for product in products:
        if 'shop' in product:
            shop = product['shop']
            if shop not in shop_products:
                shop_products[shop] = []
            shop_products[shop].append(product)
    
    # Check if any shop has more than one product for this model_id
    for shop, shop_product_list in shop_products.items():
        if len(shop_product_list) > 1:
            duplicates_in_same_store.append(model_id)
            break  # Only need to record model_id once

if duplicates_in_same_store:
    print(f"Found {len(duplicates_in_same_store)} model IDs with duplicates in the same store:")
    for model_id in sorted(duplicates_in_same_store):
        print(f"  {model_id}")
else:
    print("No duplicates found in the same store.")

