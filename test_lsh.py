import json
from collections import OrderedDict
from min_hash_lsh import get_mw, get_binary_vector, min_hash, lsh
from data_cleaning import clean_data, restructure

def get_factors(n):
    factors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
    return sorted(factors)

def get_actual_duplicates(keys):
    dup = set()
    model_map = {}
    for key in keys:
        model_id = key.split('_')[0]
        if model_id not in model_map:
            model_map[model_id] = []
        model_map[model_id].append(key)
    for model_id, keys_list in model_map.items():
        if len(keys_list) > 1:
            for i in range(len(keys_list)):
                for j in range(i + 1, len(keys_list)):
                    dup.add(tuple(sorted([keys_list[i], keys_list[j]])))
    return dup

def evaluate_lsh(dissim_df, actual_dup):
    candidates = set()
    for i, k1 in enumerate(dissim_df.index):
        for j, k2 in enumerate(dissim_df.columns):
            if i < j and dissim_df.loc[k1, k2] == 0.0:
                candidates.add(tuple(sorted([k1, k2])))
    tp = candidates & actual_dup
    fp = candidates - actual_dup
    fn = actual_dup - candidates
    prec = len(tp) / len(candidates) if candidates else 0
    rec = len(tp) / len(actual_dup) if actual_dup else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {
        'candidates': len(candidates),
        'tp': len(tp),
        'fp': len(fp),
        'fn': len(fn),
        'prec': prec,
        'rec': rec,
        'f1': f1
    }

def main():
    with open('TVs-all-merged.json', 'r', encoding='utf-8') as f:
        data = restructure(clean_data(json.load(f)))
    
    actual_dup = get_actual_duplicates(list(data.keys()))
    print(f"Products: {len(data)}, Actual duplicates: {len(actual_dup)}")
    
    mw = get_mw(data)
    signature_m = min_hash(get_binary_vector(mw, data), 0.5)
    n = signature_m.shape[0]
    print(f"n={n}\n")
    
    results = []
    for b in get_factors(n):
        r = n // b
        threshold = (1.0 / b) ** (1.0 / r)
        eval_result = evaluate_lsh(lsh(signature_m, list(data.keys()), b), actual_dup)
        eval_result.update({'b': b, 'r': r, 'threshold': threshold})
        results.append(eval_result)
        print(f"b={b}, r={r}, t={threshold:.4f}: {eval_result['candidates']} candidates, "
              f"TP={eval_result['tp']}, FP={eval_result['fp']}, FN={eval_result['fn']}, "
              f"P={eval_result['prec']:.4f}, R={eval_result['rec']:.4f}, F1={eval_result['f1']:.4f}")
    
    print("\n" + "="*80)
    print(f"{'b':<6} {'r':<6} {'threshold':<10} {'Candidates':<12} {'TP':<6} {'FP':<6} {'FN':<6} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    print("-"*80)
    for r in results:
        print(f"{r['b']:<6} {r['r']:<6} {r['threshold']:<10.4f} {r['candidates']:<12} {r['tp']:<6} "
              f"{r['fp']:<6} {r['fn']:<6} {r['prec']:<8.4f} {r['rec']:<8.4f} {r['f1']:<8.4f}")

if __name__ == "__main__":
    main()

