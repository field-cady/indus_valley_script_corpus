import pandas as pd
import re
import random

# Load mapping
mapping_df = pd.read_csv('indus_dravidian_mapping.csv')
mapping = dict(zip(mapping_df['Indus_Sign'].astype(str), mapping_df['Dravidian_Syllable']))

# Load inscriptions
df_indus = pd.read_csv('inscriptions.csv')

def translate_sample(artifact_type_pattern, num_samples, title):
    print(f"\n--- {title} ---")
    subset = df_indus[df_indus['type'].str.contains(artifact_type_pattern, case=False, na=False)].copy()
    unique_texts = subset['text'].dropna().unique()
    
    # Filter for texts that have at least some mapped signs to make it interesting
    valid_texts = []
    for text in unique_texts:
        signs = re.findall(r'\d+', text)
        mapped_count = sum(1 for s in signs if s in mapping)
        if len(signs) >= 3 and mapped_count >= len(signs) - 2:
            valid_texts.append((text, signs))
            
    random.seed(123) # For reproducibility
    sample = random.sample(valid_texts, min(num_samples, len(valid_texts)))
    
    print(f"{'Original Text':<22} | {'Signs (RTL)':<25} | {'Pronunciation'}")
    print("-" * 70)
    
    for text, signs in sample:
        signs_rtl = signs[::-1]
        pronunciation = []
        for s in signs_rtl:
            pronunciation.append(mapping.get(s, f"[{s}]"))
        
        signs_str = "-".join(signs_rtl)
        pron_str = "".join(pronunciation)
        print(f"{text:<22} | {signs_str:<25} | {pron_str}")

# Translate Tags (Shipping/Trade labels)
translate_sample(r'^TAG', 10, "Translating 10 TAGS (Shipping Labels & Sealings)")

# Translate Tablets (Tokens/Receipts)
translate_sample(r'^TAB', 10, "Translating 10 TABLETS (Mass-Produced Tokens)")

