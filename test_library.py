import pandas as pd
df = pd.read_csv("data/filtered_skintone_reviews.csv", low_memory=False)
print(df.columns.tolist())
print(df['review_text'].dropna().head(3))


df = pd.read_csv("data/filtered_skintone_reviews.csv", low_memory=False)
print(df['primary_category'].value_counts().head(10))
print(df['secondary_category'].value_counts().head(10))


df = pd.read_csv("data/filtered_skintone_reviews.csv", low_memory=False)

# Check for skin sensitivity mentions
sensitivity_terms = ['eczema', 'rash', 'breakout', 'irritat', 'sensitive', 'reaction', 'burn', 'sting', 'flare']

for term in sensitivity_terms:
    count = df['review_text'].dropna().str.lower().str.contains(term).sum()
    print(f"{term}: {count} reviews")