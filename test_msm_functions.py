"""
Test script for all MSM functions using the first 2 products from TVs-all-merged.json
"""
import json
from MSM import *

# Load the data
print("=" * 80)
print("Loading data from TVs-all-merged.json...")
print("=" * 80)
with open('TVs-all-merged.json', 'r') as f:
    data = json.load(f)

# Get the first 2 products
model_ids = list(data.keys())[:2]
p1 = data[model_ids[0]][0]
p2 = data[model_ids[1]][0]

print(f"\nProduct 1 - Model ID: {p1.get('modelID')}")
print(f"  Shop: {p1.get('shop')}")
print(f"  Title: {p1.get('title')[:70]}...")
print(f"  Features: {len(p1.get('featuresMap', {}))} attributes")
print(f"  Brand: {p1.get('featuresMap', {}).get('Brand', 'N/A')}")

print(f"\nProduct 2 - Model ID: {p2.get('modelID')}")
print(f"  Shop: {p2.get('shop')}")
print(f"  Title: {p2.get('title')[:70]}...")
print(f"  Features: {len(p2.get('featuresMap', {}))} attributes")
print(f"  Brand: {p2.get('featuresMap', {}).get('Brand', 'N/A')}")

print("\n" + "=" * 80)
print("TESTING ALL MSM FUNCTIONS")
print("=" * 80)

# Test 1: calcSim
print("\n1. Testing calcSim(s1, s2)")
print("-" * 80)
s1 = p1.get('title', '')
s2 = p2.get('title', '')
result = calcSim(s1, s2)
print(f"   String 1: {s1[:60]}...")
print(f"   String 2: {s2[:60]}...")
print(f"   Result (Q-Gram Similarity, q=3): {result:.6f}")
# Test with same string
result_same = calcSim(s1, s1)
print(f"   Same string test: {result_same:.6f} (should be 1.0)")
assert abs(result_same - 1.0) < 0.0001, "Same string should have similarity 1.0"
print("   ✓ PASSED")

# Test 2: sameShop
print("\n2. Testing sameShop(p1, p2)")
print("-" * 80)
result = sameShop(p1, p2)
print(f"   Product 1 shop: {p1.get('shop')}")
print(f"   Product 2 shop: {p2.get('shop')}")
print(f"   Result: {result}")
# Test with same product
result_same = sameShop(p1, p1)
print(f"   Same product test: {result_same} (should be True)")
assert result_same == True, "Same product should return True"
print("   ✓ PASSED")

# Test 3: diffBrand
print("\n3. Testing diffBrand(p1, p2)")
print("-" * 80)
brand1 = p1.get('featuresMap', {}).get('Brand', '')
brand2 = p2.get('featuresMap', {}).get('Brand', '')
result = diffBrand(p1, p2)
print(f"   Product 1 brand: {brand1 if brand1 else 'N/A'}")
print(f"   Product 2 brand: {brand2 if brand2 else 'N/A'}")
print(f"   Result: {result}")
# Test with same product
result_same = diffBrand(p1, p1)
print(f"   Same product test: {result_same} (should be False)")
assert result_same == False, "Same product should return False"
print("   ✓ PASSED")

# Test 4: key
print("\n4. Testing key(kvp)")
print("-" * 80)
# Test with tuple format
kvp_tuple = ("Screen Size", "32\"")
result_tuple = key(kvp_tuple)
print(f"   KVP tuple: {kvp_tuple}")
print(f"   Result: {result_tuple}")
assert result_tuple == "Screen Size", "Should extract key from tuple"
# Test with dict format
kvp_dict = {"key": "Brand", "value": "Samsung"}
result_dict = key(kvp_dict)
print(f"   KVP dict: {kvp_dict}")
print(f"   Result: {result_dict}")
assert result_dict == "Brand", "Should extract key from dict"
# Test with actual featuresMap entry
if p1.get('featuresMap'):
    first_key = list(p1['featuresMap'].keys())[0]
    first_val = p1['featuresMap'][first_key]
    kvp_from_map = (first_key, first_val)
    result_map = key(kvp_from_map)
    print(f"   KVP from featuresMap: {kvp_from_map}")
    print(f"   Result: {result_map}")
    assert result_map == first_key, "Should extract key from featuresMap entry"
print("   ✓ PASSED")

# Test 5: value
print("\n5. Testing value(kvp)")
print("-" * 80)
# Test with tuple format
kvp_tuple = ("Screen Size", "32\"")
result_tuple = value(kvp_tuple)
print(f"   KVP tuple: {kvp_tuple}")
print(f"   Result: {result_tuple}")
assert result_tuple == "32\"", "Should extract value from tuple"
# Test with dict format
kvp_dict = {"key": "Brand", "value": "Samsung"}
result_dict = value(kvp_dict)
print(f"   KVP dict: {kvp_dict}")
print(f"   Result: {result_dict}")
assert result_dict == "Samsung", "Should extract value from dict"
# Test with actual featuresMap entry
if p1.get('featuresMap'):
    first_key = list(p1['featuresMap'].keys())[0]
    first_val = p1['featuresMap'][first_key]
    kvp_from_map = (first_key, first_val)
    result_map = value(kvp_from_map)
    print(f"   KVP from featuresMap: {kvp_from_map}")
    print(f"   Result: {result_map}")
    assert result_map == first_val, "Should extract value from featuresMap entry"
print("   ✓ PASSED")

# Test 6: exMW
print("\n6. Testing exMW(product)")
print("-" * 80)
mw1 = exMW(p1)
mw2 = exMW(p2)
print(f"   Product 1 model words ({len(mw1)}): {sorted(list(mw1))}")
print(f"   Product 2 model words ({len(mw2)}): {sorted(list(mw2))}")
assert isinstance(mw1, set), "Should return a set"
assert isinstance(mw2, set), "Should return a set"
print("   ✓ PASSED")

# Test 7: mw
print("\n7. Testing mw(set_C, set_D)")
print("-" * 80)
mw1 = exMW(p1)
mw2 = exMW(p2)
result = mw(mw1, mw2)
common = mw1 & mw2
union = mw1 | mw2
print(f"   Set C (Product 1): {len(mw1)} model words")
print(f"   Set D (Product 2): {len(mw2)} model words")
print(f"   Common model words: {sorted(list(common))}")
print(f"   Union size: {len(union)}")
print(f"   Result (Jaccard similarity): {result:.6f}")
# Verify calculation
expected = len(common) / len(union) if len(union) > 0 else 0.0
assert abs(result - expected) < 0.0001, f"Jaccard calculation should be {expected}"
# Test with same sets
result_same = mw(mw1, mw1)
print(f"   Same sets test: {result_same:.6f} (should be 1.0)")
assert abs(result_same - 1.0) < 0.0001, "Same sets should have similarity 1.0"
print("   ✓ PASSED")

# Test 8: TMWMSim
print("\n8. Testing TMWMSim(p1, p2, alpha, beta)")
print("-" * 80)
# Test with different threshold combinations
alpha1, beta1 = 0.5, 0.3
result1 = TMWMSim(p1, p2, alpha1, beta1)
print(f"   Alpha={alpha1}, Beta={beta1}: {result1}")
assert result1 == 1.0 or result1 == -1.0 or (0.0 <= result1 <= 1.0), "Result should be 1.0, -1.0, or between 0 and 1"

alpha2, beta2 = 0.1, 0.1
result2 = TMWMSim(p1, p2, alpha2, beta2)
print(f"   Alpha={alpha2}, Beta={beta2}: {result2}")
assert result2 == 1.0 or result2 == -1.0 or (0.0 <= result2 <= 1.0), "Result should be 1.0, -1.0, or between 0 and 1"

alpha3, beta3 = 0.9, 0.9
result3 = TMWMSim(p1, p2, alpha3, beta3)
print(f"   Alpha={alpha3}, Beta={beta3}: {result3}")
assert result3 == 1.0 or result3 == -1.0 or (0.0 <= result3 <= 1.0), "Result should be 1.0, -1.0, or between 0 and 1"
print("   ✓ PASSED")

# Test 9: minFeatures
print("\n9. Testing minFeatures(p1, p2)")
print("-" * 80)
features1 = len(p1.get('featuresMap', {}))
features2 = len(p2.get('featuresMap', {}))
result = minFeatures(p1, p2)
expected = min(features1, features2)
print(f"   Product 1 features: {features1}")
print(f"   Product 2 features: {features2}")
print(f"   Result: {result}")
assert result == expected, f"Should return minimum: {expected}"
# Test with same product
result_same = minFeatures(p1, p1)
print(f"   Same product test: {result_same} (should be {features1})")
assert result_same == features1, "Same product should return its feature count"
print("   ✓ PASSED")

# Test 10: hClustering
print("\n10. Testing hClustering(dist, epsilon)")
print("-" * 80)
# Create a simple distance matrix for 4 items
# Items 0 and 1 are close (distance 0.1), items 2 and 3 are close (distance 0.1)
# Items 0/1 are far from items 2/3 (distance 0.9)
dist_matrix = [
    [0.0, 0.1, 0.9, 0.9],  # Item 0
    [0.1, 0.0, 0.9, 0.9],  # Item 1
    [0.9, 0.9, 0.0, 0.1],  # Item 2
    [0.9, 0.9, 0.1, 0.0]   # Item 3
]
epsilon = 0.5
clusters = hClustering(dist_matrix, epsilon)
print(f"   Distance matrix: 4 items with 2 close pairs")
print(f"   Epsilon: {epsilon}")
print(f"   Result clusters: {clusters}")
# Should have 2 clusters: {0, 1} and {2, 3}
assert len(clusters) == 2, f"Should have 2 clusters, got {len(clusters)}"
assert {0, 1} in clusters or {1, 0} in clusters, "Should cluster items 0 and 1 together"
assert {2, 3} in clusters or {3, 2} in clusters, "Should cluster items 2 and 3 together"
print("   ✓ PASSED")

# Test with smaller epsilon (should create more clusters)
print("\n   Testing with smaller epsilon (epsilon=0.05)...")
epsilon2 = 0.05
clusters2 = hClustering(dist_matrix, epsilon2)
print(f"   Epsilon: {epsilon2}")
print(f"   Result clusters: {clusters2}")
assert len(clusters2) == 4, f"With small epsilon, should have 4 clusters (one per item), got {len(clusters2)}"
print("   ✓ PASSED")

# Test with empty distance matrix
print("\n   Testing with empty distance matrix...")
clusters_empty = hClustering([], 0.5)
print(f"   Result: {clusters_empty}")
assert clusters_empty == [], "Empty matrix should return empty list"
print("   ✓ PASSED")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\nSummary:")
print(f"  - Tested all 10 MSM functions")
print(f"  - Used products: {p1.get('modelID')} and {p2.get('modelID')}")
print(f"  - All assertions passed ✓")

