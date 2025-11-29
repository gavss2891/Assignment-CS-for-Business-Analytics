import re
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import json

def replace(m, string):
    regex = {
        re.compile(rf'{perm}', re.IGNORECASE): repl
        for repl, perms in m.items()
        for perm in perms
    }
    for pattern, repl in regex.items():
        string = pattern.sub(repl, string)
    return string


def clean_data(data):
    map_mw = {
        "inch ": ["Inch", "inches", '"', "-inch", " inch", "inch"],
        "hz ": ["Hertz", "hertz", "Hz", "HZ", " hz", "-hz", "hz"]
    }
    
    for name, products in data.items():
        for product in products:
            if "title" in product and product["title"]:
                product["title"] = replace(map_mw, product["title"].lower())
            
            if "featuresMap" in product and isinstance(product["featuresMap"], dict):
                for feature, value in product["featuresMap"].items():
                    if value:
                        product["featuresMap"][feature] = replace(map_mw, value.lower())
    
    return data


def separate_duplicates(data):
    dup = OrderedDict()
    non_dup = OrderedDict()
    
    for name, products in data.items():
        if len(products) > 1:
            dup[name] = products
        else:
            non_dup[name] = products
    
    return dup, non_dup


def restructure(data):
    res = OrderedDict()
    for name, products in data.items():
        for product in products:
            shop = product.get("shop")
            if shop:
                res[f"{name}_{shop}"] = product
    return res


def test_train_split(dup, non_dup):
    dup_keys = list(dup.keys())
    non_dup_keys = list(non_dup.keys())
    
    dup_train_keys, dup_test_keys = train_test_split(
        dup_keys, train_size=0.67, random_state=42
    )
    non_dup_train_keys, non_dup_test_keys = train_test_split(
        non_dup_keys, train_size=0.67, random_state=42
    )
    
    dup_train = {key: dup[key] for key in dup_train_keys}
    dup_test = {key: dup[key] for key in dup_test_keys}
    non_dup_train = {key: non_dup[key] for key in non_dup_train_keys}
    non_dup_test = {key: non_dup[key] for key in non_dup_test_keys}
    
    train = restructure(dup_train)
    train.update(restructure(non_dup_train))
    
    test = restructure(dup_test)
    test.update(restructure(non_dup_test))
    
    return train, test


def main():
    with open('TVs-all-merged.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    data = clean_data(data)
    dup, non_dup = separate_duplicates(data)
    train, test = test_train_split(dup, non_dup)
    
    print(f"Train: {len(train)}, Test: {len(test)}")
    for k, v in list(train.items())[:20]:
        print(f"  {k}: {v.get('title', '')}")
    
    return train, test

if __name__ == "__main__":
    main()