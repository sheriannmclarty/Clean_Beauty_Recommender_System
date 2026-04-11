import pandas as pd

# Skip the 4 header rows
annex2 = pd.read_csv(
    "data/cosing_annex_prohibited_v2.txt",
    skiprows=4,
    encoding='utf-8',
    on_bad_lines='skip'
)

annex3 = pd.read_csv(
    "data/cosing_annex3_restricted_v2.txt",
    skiprows=4,
    encoding='utf-8',
    on_bad_lines='skip'
)

print("=== ANNEX 2 (Prohibited) ===")
print(annex2.shape)
print(annex2.columns.tolist())
print(annex2.head(3))

print("\n=== ANNEX 3 (Restricted) ===")
print(annex3.shape)
print(annex3.columns.tolist())
print(annex3.head(3))