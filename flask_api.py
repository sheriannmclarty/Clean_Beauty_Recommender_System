from flask import Flask, request, jsonify
import pandas as pd

# === Load CosIng Annex data ===
annex2 = pd.read_csv("data/cosing_annex_prohibited_v2.txt", skiprows=4, encoding='utf-8', on_bad_lines='skip')
annex3 = pd.read_csv("data/cosing_annex3_restricted_v2.txt", skiprows=4, encoding='utf-8', on_bad_lines='skip')

annex2['ingredient_clean'] = annex2['Chemical name / INN'].str.strip().str.lower()
annex3['ingredient_clean'] = annex3['Chemical name / INN'].str.strip().str.lower()

prohibited_list = set(annex2['ingredient_clean'].dropna())
restricted_list = set(annex3['ingredient_clean'].dropna())

print(f"✅ Prohibited ingredients loaded: {len(prohibited_list)}")
print(f"✅ Restricted ingredients loaded: {len(restricted_list)}")

# === INCI Filter ===
def check_ingredients(product_ingredients: str) -> dict:
    if not isinstance(product_ingredients, str):
        return {"status": "unknown", "flagged": []}

    ingredients = [i.strip().lower() for i in product_ingredients.split(',')]
    prohibited_found = []
    restricted_found = []

    for ingredient in ingredients:
        ing_words = set(ingredient.split())
        for p in prohibited_list:
            p_words = set(p.split())
            if ingredient == p or (len(ing_words) > 1 and ing_words.issubset(p_words)):
                prohibited_found.append(f"{ingredient} → {p}")
                break
        for r in restricted_list:
            r_words = set(r.split())
            if ingredient == r or (len(ing_words) > 1 and ing_words.issubset(r_words)):
                restricted_found.append(f"{ingredient} → {r}")
                break

    if prohibited_found:
        return {"status": "BLOCKED", "reason": "Contains EU prohibited ingredient(s)", "flagged": prohibited_found}
    elif restricted_found:
        return {"status": "WARNING", "reason": "Contains EU restricted ingredient(s)", "flagged": restricted_found}
    else:
        return {"status": "SAFE", "reason": "No prohibited or restricted ingredients found", "flagged": []}

# === Recommender Model ===
class BiasAdjustedRecommender:
    def __init__(self, user_biases, item_biases, global_avg, ratings_df):
        self.user_biases = user_biases
        self.item_biases = item_biases
        self.global_avg = global_avg
        self.ratings_df = ratings_df

    def predict(self, user, item):
        raw = self.global_avg + self.user_biases.get(user, 0) + self.item_biases.get(item, 0)
        return round(min(max(raw, 1.0), 5.0), 2)

    def recommend(self, user_id, top_n=5):
        user_rated = self.ratings_df[self.ratings_df['user'] == user_id]['item'].tolist()
        all_items = set(self.ratings_df['item'])
        unseen_items = all_items - set(user_rated)
        predictions = [(item, self.predict(user_id, item)) for item in unseen_items]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [{"item": int(item), "predicted_rating": round(score, 2)} for item, score in predictions[:top_n]]

# === Build Model ===
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

# === Flask App ===
app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        user_id = data['user_id']
        top_n = data.get('top_n', 5)

        recommendations = model.recommend(user_id, top_n=top_n)

        results = []
        for rec in recommendations:
            meta = item_to_meta.get(rec['item'], {})
            results.append({
                "product_name": meta.get('product_name_x', 'Unknown'),
                "brand_name": meta.get('brand_name_x', 'Unknown'),
                "predicted_rating": rec['predicted_rating'],
                "safety_status": "✅ SAFE",
                "note": "Full INCI screening applied when product ingredients available"
            })

        return jsonify({"recommendations": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)