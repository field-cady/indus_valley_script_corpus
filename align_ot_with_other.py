import pandas as pd
import numpy as np
import re
from collections import Counter
import ot

def run_ot_with_other(N):
    print(f"\n--- Running OT for N={N} with 'OTHER' token ---")
    
    # 1. Prepare Indus Data (Graph 1)
    df_indus = pd.read_csv('inscriptions.csv')
    seals = df_indus[df_indus['type'].str.contains('S|seal', case=False, na=False)]

    indus_sequences = []
    for text in seals['text'].dropna():
        signs = re.findall(r'\d+', text)
        # Strip 740 and 000 from the terminal position (index 0)
        while len(signs) > 0 and signs[0] in ['740', '000']:
            signs.pop(0)
        if len(signs) > 1:
            indus_sequences.append(signs)

    all_signs = [sign for seq in indus_sequences for sign in seq]
    top_signs_counts = Counter(all_signs).most_common(N)
    top_signs = [s for s, _ in top_signs_counts]
    
    # Add 'OTHER' to the vocabulary
    vocab_signs = top_signs + ['OTHER']
    sign_to_idx = {s: i for i, s in enumerate(vocab_signs)}

    C1 = np.zeros((N + 1, N + 1))
    for seq in indus_sequences:
        # Replace non-top N signs with 'OTHER'
        processed_seq = [s if s in sign_to_idx else 'OTHER' for s in seq]
        for i in range(len(processed_seq) - 1):
            s1, s2 = processed_seq[i], processed_seq[i+1]
            C1[sign_to_idx[s1], sign_to_idx[s2]] += 1

    row_sums1 = C1.sum(axis=1, keepdims=True)
    C1_prob = np.divide(C1, row_sums1, out=np.zeros_like(C1), where=row_sums1!=0)
    C1_dist = 1.0 - C1_prob


    # 2. Prepare Dravidian Data (Graph 2)
    surnames = []
    with open('dravidian_surnames_large.csv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            name = line.split(',')[0].strip()
            if name:
                surnames.append(name)

    def extract_syllables(name):
        name = name.lower()
        parts = re.findall(r'[^aeiouy]*[aeiouy]+(?:[^aeiouy]*(?=[^aeiouy]*[aeiouy]|$))?', name)
        return [p for p in parts if p]

    drav_sequences = [extract_syllables(name) for name in surnames]
    all_syls = [syl for seq in drav_sequences for syl in seq]
    top_syls_counts = Counter(all_syls).most_common(N)
    top_syls = [s for s, _ in top_syls_counts]
    
    vocab_syls = top_syls + ['OTHER']
    syl_to_idx = {s: i for i, s in enumerate(vocab_syls)}

    C2 = np.zeros((N + 1, N + 1))
    for seq in drav_sequences:
        processed_seq = [s if s in syl_to_idx else 'OTHER' for s in seq]
        for i in range(len(processed_seq) - 1):
            s1, s2 = processed_seq[i], processed_seq[i+1]
            C2[syl_to_idx[s1], syl_to_idx[s2]] += 1

    row_sums2 = C2.sum(axis=1, keepdims=True)
    C2_prob = np.divide(C2, row_sums2, out=np.zeros_like(C2), where=row_sums2!=0)
    C2_dist = 1.0 - C2_prob


    # 3. Gromov-Wasserstein Optimal Transport
    p = ot.unif(N + 1)
    q = ot.unif(N + 1)

    try:
        gw_matrix, log = ot.gromov.gromov_wasserstein(
            C1_dist, C2_dist, p, q, loss_fun='square_loss', log=True
        )
    except Exception as e:
        print(f"[-] Error in GW: {e}")
        return None, None

    # 4. Extract Mappings (ignoring 'OTHER')
    mapping = {}
    for i in range(N): # Only loop over top N signs, not 'OTHER'
        best_match_idx = np.argmax(gw_matrix[i])
        sign = vocab_signs[i]
        syl = vocab_syls[best_match_idx]
        mapping[sign] = syl
        
    return mapping, top_signs

map_50, top_50_signs = run_ot_with_other(50)
map_100, top_100_signs = run_ot_with_other(100)

if map_50 and map_100:
    common_signs = set(top_50_signs).intersection(set(top_100_signs))
    
    exact_matches = 0
    print("\n--- Comparison (N=50 vs N=100) with 'OTHER' handling ---")
    for sign in common_signs:
        syl_50 = map_50[sign]
        syl_100 = map_100[sign]
        if syl_50 == syl_100:
            exact_matches += 1
            print(f"EXACT MATCH: Sign {sign} -> '{syl_50}'")
            
    similarity = (exact_matches / len(common_signs)) * 100
    print(f"\nTotal common signs compared: {len(common_signs)}")
    print(f"Total exact matches: {exact_matches}")
    print(f"Similarity: {similarity:.2f}%")
    
    print("\nHow the Top 5 Signs Shifted:")
    top_5 = top_50_signs[:5]
    for sign in top_5:
        print(f"Sign {sign}: '{map_50.get(sign)}' (N=50) -> '{map_100.get(sign)}' (N=100)")
