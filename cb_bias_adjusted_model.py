import pandas as pd
import pickle

# === Step 1: Load and preprocess data ===
df = pd.read_csv("data/filtered_skintone_reviews.csv", dtype={"product_name_x": str, "brand_name_x": str}, low_memory=False)
df = df.dropna(subset=['author_id', 'product_name_x', 'brand_name_x', 'rating_x'])
df['rating_x'] = pd.to_numeric(df['rating_x'], errors='coerce')

# Encode users and items
df['user'] = df['author_id'].astype('category').cat.codes
df['item'] = df['product_name_x'].astype('category').cat.codes

# === Step 2: Compute biases ===
global_avg = df['rating_x'].mean()
user_bias = (df.groupby('user')['rating_x'].mean() - global_avg).to_dict()
item_bias = (df.groupby('item')['rating_x'].mean() - global_avg).to_dict()

# === Step 3: Define the model ===
class BiasAdjustedRecommender:
    def __init__(self, user_biases, item_biases, global_avg, ratings_df):
        self.user_biases = user_biases
        self.item_biases = item_biases
        self.global_avg = global_avg
        self.ratings_df = ratings_df

    def predict(self, user, item):
        raw = self.global_avg + self.user_biases.get(user, 0) + self.item_biases.get(item, 0)
        return round(min(max(raw, 1.0), 5.0), 2)  # ✅ clipped between 1-5

    def recommend(self, user_id, top_n=5):
        user_rated = self.ratings_df[self.ratings_df['user'] == user_id]['item'].tolist()
        all_items = set(self.ratings_df['item'])
        unseen_items = all_items - set(user_rated)

        # Count reviews per item for popularity weighting
        item_review_counts = self.ratings_df['item'].value_counts().to_dict()

        predictions = []
        for item in unseen_items:
            base_score = self.predict(user_id, item)
            # Add small popularity boost to break ties
            popularity = item_review_counts.get(item, 1)
            popularity_factor = round(min(popularity / 1000, 0.05), 4)
            final_score = round(min(base_score + popularity_factor, 5.0), 2)
            predictions.append((item, final_score))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return [{"item": int(item), "predicted_rating": round(score, 2)}
                for item, score in predictions[:top_n]]
# === Step 4: Create metadata dict ===
item_to_meta = df.drop_duplicates(subset='item').set_index('item')[['product_name_x', 'brand_name_x']].to_dict(orient='index')

# === Step 5: Save model and metadata ===
model = BiasAdjustedRecommender(user_bias, item_bias, global_avg, df)
with open("cb_model.pkl", "wb") as f:
    pickle.dump((model, item_to_meta), f)

print("✅ Model and item metadata saved to cb_model.pkl")
