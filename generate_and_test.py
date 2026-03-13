import pandas as pd
import numpy as np
import re
from collections import Counter
import ot
import random

# Settings
N = 50  # Top N signs and syllables to match

print(f"1. Generating full mapping for top {N} signs...")

# --- 1. OT MAPPING PIPELINE ---
df_indus = pd.read_csv('inscriptions.csv')
seals = df_indus[df_indus['type'].str.contains('S|seal', case=False, na=False)]

indus_sequences = []
for text in seals['text'].dropna():
    signs = re.findall(r'\d+', text)
    while len(signs) > 0 and signs[0] in ['740', '000']:
        signs.pop(0)
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

row_sums1 = C1.sum(axis=1, keepdims=True)
C1_prob = np.divide(C1, row_sums1, out=np.zeros_like(C1), where=row_sums1!=0)
C1_dist = 1.0 - C1_prob

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

row_sums2 = C2.sum(axis=1, keepdims=True)
C2_prob = np.divide(C2, row_sums2, out=np.zeros_like(C2), where=row_sums2!=0)
C2_dist = 1.0 - C2_prob

p = ot.unif(N)
q = ot.unif(N)
gw_matrix, log = ot.gromov.gromov_wasserstein(C1_dist, C2_dist, p, q, loss_fun='square_loss', log=True)

# Create dictionary mapping
mapping = {}
mapping_records = []
for i in range(N):
    best_match_idx = np.argmax(gw_matrix[i])
    sign = top_signs[i]
    syl = top_syls[best_match_idx]
    mapping[sign] = syl
    mapping_records.append({'Indus_Sign': sign, 'Dravidian_Syllable': syl})

# Save to CSV
pd.DataFrame(mapping_records).to_csv('indus_dravidian_mapping.csv', index=False)
print("Mapping saved to 'indus_dravidian_mapping.csv'\n")

# --- 2. CONTAINER TEST ---
print("2. Testing 20 Container Inscriptions...")
containers = df_indus[df_indus['type'].str.contains('POT', case=False, na=False)].copy()
unique_container_texts = containers['text'].dropna().unique()

# Filter out very short ones or ones that consist solely of unmapped signs for a better test
valid_texts = []
for text in unique_container_texts:
    signs = re.findall(r'\d+', text)
    mapped_count = sum(1 for s in signs if s in mapping)
    if len(signs) >= 2 and mapped_count >= len(signs) - 1: # mostly mapped
        valid_texts.append(text)

random.seed(42) # for reproducibility
sample_texts = random.sample(valid_texts, min(20, len(valid_texts)))

print(f"\n{'Original Text':<20} | {'Signs (RTL)':<30} | {'Pronunciation (RTL/Transliterated)'}")
print("-" * 80)

for text in sample_texts:
    signs = re.findall(r'\d+', text)
    # The signs in the text are written LTR as transcribed, but the script is RTL.
    # We will reverse the list of signs to get the true phonetic sequence from start to finish.
    signs_rtl = signs[::-1] 
    
    pronunciation = []
    for s in signs_rtl:
        if s in mapping:
            pronunciation.append(mapping[s])
        else:
            pronunciation.append(f"[{s}]") # Unmapped sign
            
    signs_str = "-".join(signs_rtl)
    pron_str = "".join(pronunciation)
    print(f"{text:<20} | {signs_str:<30} | {pron_str}")

