import requests
import pandas as pd

# === Load and prepare data ===
df = pd.read_csv("data/filtered_skintone_reviews.csv", low_memory=False)
df = df.dropna(subset=['author_id', 'product_name_x', 'rating_x'])
df['author_id'] = df['author_id'].astype(str)
df['user'] = df['author_id'].astype('category').cat.codes

# === Find most active melanin-rich user ===
melanin_tones = ['deep', 'rich', 'ebony', 'dark', 'deeptan', 'deep tan']
melanin_users = df[df['skin_tone'].str.lower().isin(melanin_tones)]

if melanin_users.empty:
    print("⚠️ No melanin-rich users found.")
    print("Available skin tones:", df['skin_tone'].unique())
else:
    top_user = melanin_users['author_id'].value_counts().idxmax()
    user_row = df[df['author_id'] == top_user].iloc[0]

    print(f"✅ Selected user: {top_user}")
    print(f"   Skin tone: {user_row['skin_tone']}")
    print(f"   Number of reviews: {melanin_users['author_id'].value_counts()[top_user]}")

    encoded_id = df[df['author_id'] == top_user]['user'].iloc[0]

    # === Send request to Flask API ===
    url = "http://127.0.0.1:5000/recommend"
    payload = {"user_id": int(encoded_id), "top_n": 5}
    response = requests.post(url, json=payload)

    print("\nStatus Code:", response.status_code)
    try:
        data = response.json()
        print("\n🎯 Top 5 Recommendations for Melanin-Rich User:")
        print(f"{'#':<4} {'Product':<45} {'Brand':<25} {'Predicted Rating'}")
        print("-" * 90)
        for i, rec in enumerate(data['recommendations'], 1):
            print(f"{i:<4} {rec['product_name']:<45} {rec['brand_name']:<25} {rec['predicted_rating']}")
    except Exception as e:
        print("❌ Error decoding JSON:", e)
        print("Raw text:", response.text)
