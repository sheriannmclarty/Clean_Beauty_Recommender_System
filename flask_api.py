from flask import Flask, request, jsonify
import pandas as pd

# === Define model class ===
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
        predictions = [(item, self.predict(user_id, item)) for item in unseen_items]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [{"item": int(item), "predicted_rating": round(score, 2)} for item, score in predictions[:top_n]]

# === Build model directly from data ===
df = pd.read_csv("data/filtered_skintone_reviews.csv", low_memory=False, dtype={"product_name_x": str})
df = df.dropna(subset=['author_id', 'product_name_x', 'rating_x'])
df['rating_x'] = pd.to_numeric(df['rating_x'], errors='coerce')
df['user'] = df['author_id'].astype('category').cat.codes
df['item'] = df['product_name_x'].astype('category').cat.codes

global_avg = df['rating_x'].mean()
user_bias = (df.groupby('user')['rating_x'].mean() - global_avg).to_dict()
item_bias = (df.groupby('item')['rating_x'].mean() - global_avg).to_dict()
item_to_meta = df.drop_duplicates(subset='item').set_index('item')[['product_name_x', 'brand_name_x']].to_dict(orient='index')

model = BiasAdjustedRecommender(user_bias, item_bias, global_avg, df)
print("✅ Model built successfully")

# === Flask app ===
app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        user_id = data['user_id']
        top_n = data.get('top_n', 5)

        recommendations = model.recommend(user_id, top_n=top_n)

        for rec in recommendations:
            meta = item_to_meta.get(rec['item'], {})
            rec['product_name'] = meta.get('product_name_x', 'Unknown')
            rec['brand_name'] = meta.get('brand_name_x', 'Unknown')

        return jsonify({"recommendations": recommendations})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
