import nlpaug.augmenter.word as naw
import pandas as pd
import random

syn_aug = naw.SynonymAug(aug_src='wordnet', aug_max=1)

def augment_data(df, text_col='Resume', label_col='Category', augment_frac=0.1):
    augmented_rows = []
    for _, row in df.iterrows():
        text, label = row[text_col], row[label_col]
        if random.random() < augment_frac:
            try:
                new_text = syn_aug.augment(text)
                augmented_rows.append({text_col: new_text, label_col: label})
            except Exception:
                pass
            
    aug_df = pd.DataFrame(augmented_rows)
    return pd.concat([df, aug_df], ignore_index=True)
