import pandas as pd
df = pd.read_csv("data/filtered_skintone_reviews.csv", low_memory=False)
print(f"Rows: {len(df)}")
print(f"Unique users: {df['author_id'].nunique()}")
print(f"Unique products: {df['product_name_x'].nunique()}")
