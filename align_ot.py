import pandas as pd
import numpy as np
import re
from collections import Counter
import ot

# Settings
N = 50  # Top N signs and syllables to match

print(f"Running Gromov-Wasserstein OT on top {N} Indus signs vs {N} Dravidian syllables...\n")

# 1. Prepare Indus Data (Graph 1)
df_indus = pd.read_csv('inscriptions.csv')
seals = df_indus[df_indus['type'].str.contains('S|seal', case=False, na=False)]

indus_sequences = []
for text in seals['text'].dropna():
    signs = re.findall(r'\d+', text)
    if len(signs) > 1:
        indus_sequences.append(signs)

all_signs = [sign for seq in indus_sequences for sign in seq]
top_signs = [s for s, _ in Counter(all_signs).most_common(N)]
sign_to_idx = {s: i for i, s in enumerate(top_signs)}

C1 = np.zeros((N, N))
for seq in indus_sequences:
    for i in range(len(seq) - 1):
        s1, s2 = seq[i], seq[i+1]
        if s1 in sign_to_idx and s2 in sign_to_idx:
            C1[sign_to_idx[s1], sign_to_idx[s2]] += 1

# Normalize rows to get probabilities
row_sums1 = C1.sum(axis=1, keepdims=True)
C1_prob = np.divide(C1, row_sums1, out=np.zeros_like(C1), where=row_sums1!=0)
# Convert to distance: 1 - prob
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
    # Simple phonetic split (vowel clusters + leading consonants)
    parts = re.findall(r'[^aeiouy]*[aeiouy]+(?:[^aeiouy]*(?=[^aeiouy]*[aeiouy]|$))?', name)
    return [p for p in parts if p]

drav_sequences = [extract_syllables(name) for name in surnames]
all_syls = [syl for seq in drav_sequences for syl in seq]
top_syls = [s for s, _ in Counter(all_syls).most_common(N)]
syl_to_idx = {s: i for i, s in enumerate(top_syls)}

C2 = np.zeros((N, N))
for seq in drav_sequences:
    for i in range(len(seq) - 1):
        s1, s2 = seq[i], seq[i+1]
        if s1 in syl_to_idx and s2 in syl_to_idx:
            C2[syl_to_idx[s1], syl_to_idx[s2]] += 1

row_sums2 = C2.sum(axis=1, keepdims=True)
C2_prob = np.divide(C2, row_sums2, out=np.zeros_like(C2), where=row_sums2!=0)
C2_dist = 1.0 - C2_prob


# 3. Gromov-Wasserstein Optimal Transport
# Define uniform distributions (p for Graph 1, q for Graph 2)
p = ot.unif(N)
q = ot.unif(N)

# Compute GW coupling (We leave it directed/asymmetric to preserve prefix/suffix logic)
try:
    gw_matrix, log = ot.gromov.gromov_wasserstein(
        C1_dist, C2_dist, p, q, loss_fun='square_loss', log=True
    )
    print("[+] Optimal Transport mapping completed successfully.\n")
except Exception as e:
    print(f"[-] Error in GW: {e}")
    exit(1)


# 4. Extract Top Mappings
print("--- Top 15 Indus Sign to Dravidian Syllable Mappings ---")
print(f"{'Indus Sign':<15} | {'Mapped Syllable':<20} | {'Transport Mass'}")
print("-" * 60)

# Sort by the most frequent signs first
for i in range(15):
    best_match_idx = np.argmax(gw_matrix[i])
    mass = gw_matrix[i, best_match_idx]
    
    sign = top_signs[i]
    syl = top_syls[best_match_idx]
    
    # To format nicely
    padding = ' ' * (20 - len(syl) - 2)
    print(f"Sign {sign:<10} | '{syl}'{padding} | {mass:.4f}")
