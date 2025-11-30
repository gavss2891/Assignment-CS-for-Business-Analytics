import re
import math
import Levenshtein
from ordered_set import OrderedSet

def cos_sim(s1, s2):
    set1 = set(s1.split())
    set2 = set(s2.split())
    num = len(set1 & set2)
    denom = math.sqrt(len(set1)) * math.sqrt(len(set2))
    return num/denom if denom != 0 else 0

def get_mw(source, from_title=False):
    if from_title:
        pattern = r'([a-zA-Z0-9]*((\d*\.)?\d+[^0-9, ]+)[a-zA-Z0-9]*)'
        matches = re.findall(pattern, source)
        return [m[0] for m in matches]
    else:
        features, keys = source
        pattern = r'^\d+\.\d+|\b\d+:\d+\b|(?<!\S)\d{3,}\s*[x]\s*\d{3,}(?!\S)'
        mw = OrderedSet()
        for key in keys:
            if key in features: 
                matches = re.findall(pattern, features.get(key, ""))
                mw.update(matches)
        return mw

def parse_model_word(s):
    non_numeric = ''.join(ch for ch in s if not ch.isdigit())
    numeric = ''.join(ch for ch in s if ch.isdigit())
    return non_numeric, numeric

def Levenshtein_sim(s1, s2):
    D = Levenshtein.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 0 if max_len == 0 else 1 - (D / max_len)

def avg_Levenshtein_sim(mw1_data, mw2_data, LvSimMW, approx):
    num = 0.0
    denom = 0.0
    for (w1, nn1, num1) in mw1_data:
        for (w2, nn2, num2) in mw2_data:
            if LvSimMW:
                fam_sim = Levenshtein_sim(nn1, nn2)
                if fam_sim <= approx or num1 != num2:
                    continue

            sim = Levenshtein_sim(w1, w2)
            weight = len(w1) + len(w2)
            num += sim * weight 
            denom += weight

    return num / denom if denom != 0 else 0.0

def TMWMSim(t1, t2, alpha, beta, delta, approx):
    t1 = t1.lower()
    t2 = t2.lower()
    cos_val = cos_sim(t1, t2)
    if cos_val > alpha:
        return 1
    
    mw1 = list(dict.fromkeys(get_mw(t1, from_title=True)))
    mw2 = list(dict.fromkeys(get_mw(t2, from_title=True)))
    
    if len(mw1) == 0 and len(mw2) == 0:
        return cos_val
    
    mw1_data = [(mw, *parse_model_word(mw)) for mw in mw1]
    mw2_data = [(mw, *parse_model_word(mw)) for mw in mw2]
    
    similar = False
    for (_, nn1, num1) in mw1_data:
        for (_, nn2, num2) in mw2_data:
            sim = Levenshtein_sim(nn1, nn2)
            if sim > approx and num1 != num2:
               return -1
            if sim > approx and num1 == num2:
               similar = True
    
    base_sim = avg_Levenshtein_sim(mw1_data, mw2_data, LvSimMW=False, approx=approx)
    final = beta * cos_val + (1 - beta) * base_sim
    
    if similar:
        mw_sim = avg_Levenshtein_sim(mw1_data, mw2_data, LvSimMW=True, approx=approx)
        final = delta * mw_sim + (1 - delta) * final
    
    return final

