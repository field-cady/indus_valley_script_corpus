import pandas as pd
import numpy as np
import re
from collections import Counter
from scipy.linalg import svd, orthogonal_procrustes
from scipy.spatial.distance import cdist

# Settings
N = 50  # Top N signs and syllables to match
DIM = 15 # Dimensionality of the embeddings

print(f"Running Orthogonal Procrustes Alignment on top {N} signs/syllables...")

# --- 1. Process Indus Data ---
df_indus = pd.read_csv('inscriptions.csv')
seals = df_indus[df_indus['type'].str.contains('S|seal', case=False, na=False)]

indus_sequences = []
for text in seals['text'].dropna():
    signs = re.findall(r'\d+', text)
    while len(signs) > 0 and signs[0] in ['740', '000']:
        signs.pop(0)
    if len(signs) > 1:
        indus_sequences.append(signs)

all_signs = [s for seq in indus_sequences for s in seq]
top_signs = [s for s, _ in Counter(all_signs).most_common(N)]
sign_to_idx = {s: i for i, s in enumerate(top_signs)}

# Build co-occurrence matrix (window size 2)
C1 = np.zeros((N, N))
for seq in indus_sequences:
    for i in range(len(seq) - 1):
        s1, s2 = seq[i], seq[i+1]
        if s1 in sign_to_idx and s2 in sign_to_idx:
            C1[sign_to_idx[s1], sign_to_idx[s2]] += 1
            C1[sign_to_idx[s2], sign_to_idx[s1]] += 1 # Make it symmetric for embeddings

# Convert to PMI (Pointwise Mutual Information)
sum1 = np.sum(C1)
P1 = C1 / sum1
P1_row = np.sum(P1, axis=1)
P1_col = np.sum(P1, axis=0)
PMI1 = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if P1[i, j] > 0:
            PMI1[i, j] = max(np.log2(P1[i, j] / (P1_row[i] * P1_col[j])), 0)

# SVD for Embeddings
U1, S1, V1 = svd(PMI1)
embed1 = U1[:, :DIM] * np.sqrt(S1[:DIM])
# Normalize embeddings
embed1_norm = embed1 / np.linalg.norm(embed1, axis=1, keepdims=True)
np.nan_to_num(embed1_norm, copy=False)


# --- 2. Process Dravidian Data ---
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
top_syls = [s for s, _ in Counter(all_syls).most_common(N)]
syl_to_idx = {s: i for i, s in enumerate(top_syls)}

C2 = np.zeros((N, N))
for seq in drav_sequences:
    for i in range(len(seq) - 1):
        s1, s2 = seq[i], seq[i+1]
        if s1 in syl_to_idx and s2 in syl_to_idx:
            C2[syl_to_idx[s1], syl_to_idx[s2]] += 1
            C2[syl_to_idx[s2], syl_to_idx[s1]] += 1

sum2 = np.sum(C2)
P2 = C2 / sum2
P2_row = np.sum(P2, axis=1)
P2_col = np.sum(P2, axis=0)
PMI2 = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if P2[i, j] > 0:
            PMI2[i, j] = max(np.log2(P2[i, j] / (P2_row[i] * P2_col[j])), 0)

U2, S2, V2 = svd(PMI2)
embed2 = U2[:, :DIM] * np.sqrt(S2[:DIM])
embed2_norm = embed2 / np.linalg.norm(embed2, axis=1, keepdims=True)
np.nan_to_num(embed2_norm, copy=False)


# --- 3. Orthogonal Procrustes Alignment ---
# We want to find rotation matrix R such that embed1_norm * R ~ embed2_norm
R, scale = orthogonal_procrustes(embed1_norm, embed2_norm)
embed1_rotated = np.dot(embed1_norm, R)

# --- 4. Nearest Neighbor Mapping ---
# Find closest Dravidian syllable for each Indus sign in the shared space
distances = cdist(embed1_rotated, embed2_norm, metric='cosine')

print("\n--- Top 15 Indus Sign to Dravidian Syllable Mappings (Procrustes) ---")
print(f"{'Indus Sign':<15} | {'Mapped Syllable':<20} | {'Cosine Distance'}")
print("-" * 60)

for i in range(15):
    best_match_idx = np.argmin(distances[i])
    dist = distances[i, best_match_idx]
    
    sign = top_signs[i]
    syl = top_syls[best_match_idx]
    
    padding = ' ' * (20 - len(syl) - 2)
    print(f"Sign {sign:<10} | '{syl}'{padding} | {dist:.4f}")

