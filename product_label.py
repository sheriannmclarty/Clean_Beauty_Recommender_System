
import pandas as pd
products_df = pd.read_csv("data/product_info.csv", low_memory=False)
print(products_df['primary_category'].value_counts())
print("\n")
print(products_df['secondary_category'].value_counts().head(20))