import requests
import pandas as pd
import random

# === Ethnically diverse name pool ===
ethnic_first_names = [
    'Aaliyah', 'Imani', 'Zuri', 'Keisha', 'Latoya', 'Ebony',
    'Destiny', 'Jasmine', 'Tamara', 'Shayla', 'Monique', 'Tiana',
    'Amara', 'Chioma', 'Fatima', 'Adaeze', 'Ngozi', 'Nia', 'Zainab',
    'Aisha', 'Blessing', 'Chiamaka', 'Folake', 'Yetunde',
    'Gabriela', 'Valentina', 'Marisol', 'Xiomara', 'Catalina', 'Lucia',
    'Esperanza', 'Selena', 'Yolanda', 'Rosa', 'Carmen', 'Milagros',
    'Priya', 'Ananya', 'Divya', 'Meena', 'Kavya', 'Shreya', 'Nisha'
]

ethnic_last_names = [
    'Washington', 'Jackson', 'Williams', 'Thompson', 'Robinson',
    'Okafor', 'Adeyemi', 'Nwosu', 'Mensah', 'Diallo', 'Traore',
    'Rodriguez', 'Martinez', 'Flores', 'Rivera', 'Reyes',
    'Patel', 'Sharma', 'Krishnan', 'Iyer', 'Nair'
]

def generate_ethnic_name():
    return f"{random.choice(ethnic_first_names)} {random.choice(ethnic_last_names)}"

# === Nut allergens ===
nut_allergens = [
    'shea butter', 'argan oil', 'almond oil', 'sweet almond oil',
    'macadamia oil', 'walnut oil', 'hazelnut oil', 'cashew',
    'brazil nut', 'pecan', 'pistachio', 'coconut oil',
    'coconut acid', 'coconut water', 'cocos nucifera',      # ← add these
    'butyrospermum parkii', 'argania spinosa', 'prunus amygdalus',
    'carthamus tinctorius'                                   # ← safflower (tree)
]
# === Product categories ===
categories = {
    "1": "moisturizer",
    "2": "serum",
    "3": "foundation",
    "4": "cleanser",
    "5": "sunscreen"
}

# === Load reviews data ===
df = pd.read_csv("data/filtered_skintone_reviews.csv", low_memory=False)
df = df.dropna(subset=['author_id', 'product_name_x', 'rating_x'])
df['author_id'] = df['author_id'].astype(str)
df['user'] = df['author_id'].astype('category').cat.codes

# === Load Sephora products with ingredients ===
products_df = pd.read_csv("data/product_info.csv", low_memory=False)
products_df['product_name_clean'] = products_df['product_name'].str.strip().str.lower()
print(f"✅ Products loaded: {len(products_df)} items with ingredient data")

# === Ask user preferences ===
print("\n🌿 Welcome to the Clean Beauty Recommender for Melanin-Rich Skin!\n")
print("What are you looking for?")
for key, val in categories.items():
    print(f"  {key}. {val.capitalize()}")
category_choice = input("\nEnter number (or press Enter to skip): ").strip()
selected_category = categories.get(category_choice, None)

nut_allergy = input("\nDo you have a nut allergy? (yes/no): ").strip().lower()
has_nut_allergy = nut_allergy == 'yes'

print(f"\n✅ Preferences saved!")
if selected_category:
    print(f"   Looking for: {selected_category.capitalize()}")
if has_nut_allergy:
    print(f"   ⚠️ Nut allergy filter: ON")

# === Find melanin-rich users ===
melanin_tones = ['deep', 'rich', 'ebony', 'dark', 'deeptan', 'deep tan']
melanin_users = df[df['skin_tone'].str.lower().isin(melanin_tones)]

if melanin_users.empty:
    print("⚠️ No melanin-rich users found.")
    print("Available skin tones:", df['skin_tone'].unique())
else:
    # Pick randomly from top 20 most active melanin-rich users
    top_users = melanin_users['author_id'].value_counts().head(20).index.tolist()
    top_user = random.choice(top_users)
    user_row = df[df['author_id'] == top_user].iloc[0]
    display_name = generate_ethnic_name()

    print(f"\n✅ Selected user: {display_name}")
    print(f"   Skin tone: {user_row['skin_tone']}")
    print(f"   Number of reviews: {melanin_users['author_id'].value_counts()[top_user]}")

    encoded_id = df[df['author_id'] == top_user]['user'].iloc[0]

    # === Send request to Flask API ===
    url = "http://127.0.0.1:5000/recommend"
    payload = {"user_id": int(encoded_id), "top_n": 20}
    response = requests.post(url, json=payload)

    print("\nStatus Code:", response.status_code)
    try:
        data = response.json()
        recommendations = data['recommendations']

        # === Filter by category ===
        if selected_category:
            recommendations = [
                r for r in recommendations
                if selected_category.lower() in r['product_name'].lower()
            ]

        # === Merge with product ingredients ===
        for rec in recommendations:
            product_match = products_df[
                products_df['product_name_clean'] == rec['product_name'].strip().lower()
            ]
            if not product_match.empty:
                rec['ingredients'] = product_match.iloc[0]['ingredients']
            else:
                rec['ingredients'] = 'Ingredients not available'

        # === Filter by nut allergy ===
        if has_nut_allergy:
            safe_recs = []
            flagged_count = 0
            for rec in recommendations:
                ingredients_lower = str(rec.get('ingredients', '')).lower()
                found_allergen = next(
                    (a for a in nut_allergens if a in ingredients_lower), None
                )
                if found_allergen:
                    flagged_count += 1
                else:
                    safe_recs.append(rec)
            recommendations = safe_recs
            if flagged_count > 0:
                print(f"\n⚠️ {flagged_count} product(s) removed due to nut allergen detection")

        # === Display results ===
        print(f"\n🎯 Top Recommendations for {display_name}:")
        print("-" * 90)
        if recommendations:
            for i, rec in enumerate(recommendations[:10], 1):
                print(f"\n{i}. {rec['product_name']} — {rec['brand_name']}")
                print(f"   ⭐ Predicted Rating: {rec['predicted_rating']}")
                print(f"   🧴 Ingredients: {str(rec.get('ingredients', 'N/A'))[:200]}...")
        else:
            print("No matching products found for your preferences.")

    except Exception as e:
        print("❌ Error decoding JSON:", e)
        print("Raw text:", response.text)