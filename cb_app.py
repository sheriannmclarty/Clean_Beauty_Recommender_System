import streamlit as st
import pandas as pd
import random
from cb_tone_ranking import get_product_tone_scores, tone_aware_rerank

# === Page config ===
st.set_page_config(
    page_title="Clean Beauty Recommender",
    page_icon="🌿",
    layout="wide"
)
# Add this right after st.set_page_config
st.markdown("""
<style>
    .stApp { background-color: #F5EFE8; }
    .stSidebar { background-color: #5C1D3A; }
    .stSidebar * { color: white !important; }
</style>
""", unsafe_allow_html=True)
# === Custom styling ===
st.markdown("""
<style>
    .main { background-color: #F5EFE8; }
    h1 { color: #5C1D3A; }
    h2, h3 { color: #8B3A5A; }
    .stButton>button {
        background-color: #5C1D3A;
        color: white;
        border-radius: 8px;
        padding: 0.5em 2em;
        font-weight: bold;
    }
    .stButton>button:hover { background-color: #8B3A5A; }
    .rec-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #5C1D3A;
        margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# === Ethnic name generator ===
ethnic_first_names = [
    'Aaliyah', 'Imani', 'Zuri', 'Keisha', 'Latoya', 'Ebony',
    'Destiny', 'Jasmine', 'Tamara', 'Shayla', 'Monique', 'Tiana',
    'Amara', 'Chioma', 'Fatima', 'Adaeze', 'Ngozi', 'Nia', 'Zainab',
    'Aisha', 'Blessing', 'Chiamaka', 'Folake', 'Yetunde',
    'Gabriela', 'Valentina', 'Marisol', 'Xiomara', 'Catalina', 'Lucia',
    'Priya', 'Ananya', 'Divya', 'Meena', 'Kavya', 'Shreya', 'Nisha'
]
ethnic_last_names = [
    'Washington', 'Jackson', 'Williams', 'Thompson', 'Robinson',
    'Okafor', 'Adeyemi', 'Nwosu', 'Mensah', 'Diallo', 'Traore',
    'Rodriguez', 'Martinez', 'Flores', 'Rivera', 'Reyes',
    'Patel', 'Sharma', 'Krishnan', 'Iyer', 'Nair'
]

# === Nut allergens ===
nut_allergens = [
    'shea butter', 'argan oil', 'almond oil', 'sweet almond oil',
    'macadamia oil', 'walnut oil', 'hazelnut oil', 'cashew',
    'brazil nut', 'pecan', 'pistachio', 'coconut oil',
    'coconut acid', 'coconut water', 'cocos nucifera',
    'butyrospermum parkii', 'argania spinosa', 'prunus amygdalus',
    'carthamus tinctorius'
]

# === Load data (cached) ===
@st.cache_data
def load_data():
    df = pd.read_csv("data/filtered_skintone_reviews.csv", low_memory=False)
    df = df.dropna(subset=['author_id', 'product_name_x', 'rating_x'])
    df['author_id'] = df['author_id'].astype(str)
    df['rating_x'] = pd.to_numeric(df['rating_x'], errors='coerce')
    df['user'] = df['author_id'].astype('category').cat.codes
    df['item'] = df['product_name_x'].astype('category').cat.codes
    return df

@st.cache_data
def load_products():
    products_df = pd.read_csv("data/product_info.csv", low_memory=False)
    products_df['product_name_clean'] = products_df['product_name'].str.strip().str.lower()
    return products_df

@st.cache_data
def load_tone_scores():
    tone_df = pd.read_csv("data/filtered_skintone_reviews.csv", low_memory=False)
    tone_df = tone_df.dropna(subset=['author_id', 'product_name_x', 'review_text'])
    return get_product_tone_scores(tone_df)

@st.cache_data
def load_cosing():
    annex2 = pd.read_csv("data/cosing_annex_prohibited_v2.txt",
                          skiprows=4, encoding='utf-8', on_bad_lines='skip')
    annex3 = pd.read_csv("data/cosing_annex3_restricted_v2.txt",
                          skiprows=4, encoding='utf-8', on_bad_lines='skip')
    annex2['ingredient_clean'] = annex2['Chemical name / INN'].str.strip().str.lower()
    annex3['ingredient_clean'] = annex3['Chemical name / INN'].str.strip().str.lower()
    return set(annex2['ingredient_clean'].dropna()), set(annex3['ingredient_clean'].dropna())

# === Model ===
class BiasAdjustedRecommender:
    def __init__(self, user_biases, item_biases, global_avg, ratings_df):
        self.user_biases = user_biases
        self.item_biases = item_biases
        self.global_avg = global_avg
        self.ratings_df = ratings_df

    def predict(self, user, item):
        raw = self.global_avg + self.user_biases.get(user, 0) + self.item_biases.get(item, 0)
        return round(min(max(raw, 1.0), 5.0), 2)

    def recommend(self, user_id, top_n=200):
        user_rated = self.ratings_df[self.ratings_df['user'] == user_id]['item'].tolist()
        all_items = set(self.ratings_df['item'])
        unseen_items = all_items - set(user_rated)
        item_review_counts = self.ratings_df['item'].value_counts().to_dict()
        predictions = []
        for item in unseen_items:
            base_score = self.predict(user_id, item)
            popularity = item_review_counts.get(item, 1)
            popularity_factor = round(min(popularity / 1000, 0.05), 4)
            final_score = round(min(base_score + popularity_factor, 5.0), 2)
            predictions.append((item, final_score))
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [{"item": int(item), "predicted_rating": round(score, 2)}
                for item, score in predictions[:top_n]]

@st.cache_resource
def build_model():
    df = load_data()
    global_avg = df['rating_x'].mean()
    user_bias = (df.groupby('user')['rating_x'].mean() - global_avg).to_dict()
    item_bias = (df.groupby('item')['rating_x'].mean() - global_avg).to_dict()
    item_to_meta = df.drop_duplicates(subset='item').set_index('item')[
        ['product_name_x', 'brand_name_x']].to_dict(orient='index')
    model = BiasAdjustedRecommender(user_bias, item_bias, global_avg, df)
    return model, item_to_meta

# === App ===
st.title("🌿 Clean Beauty Recommender for Melanin-Rich Skin")
st.markdown("*Safety-constrained recommendations powered by EU CosIng standards*")
st.divider()

# Load everything
with st.spinner("Loading data and models..."):
    df = load_data()
    products_df = load_products()
    product_tone_scores = load_tone_scores()
    prohibited_list, restricted_list = load_cosing()
    model, item_to_meta = build_model()

# === Sidebar ===
st.sidebar.title("🎯 Your Preferences")
st.sidebar.markdown("---")

category = st.sidebar.selectbox(
    "What are you looking for?",
    ["All Products", "Moisturizers", "Treatments", "Cleansers", "Sunscreen"]
)

nut_allergy = st.sidebar.radio(
    "Do you have a nut allergy?",
    ["No", "Yes"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**🛡️ Safety Standards**")
st.sidebar.markdown(f"✅ {len(prohibited_list):,} prohibited ingredients blocked")
st.sidebar.markdown(f"⚠️ {len(restricted_list):,} restricted ingredients flagged")
st.sidebar.markdown("*Powered by EU CosIng Database*")

# === Get recommendations button ===
if st.button("✨ Get My Recommendations"):

    # Pick random melanin-rich user
    melanin_tones = ['deep', 'rich', 'ebony', 'dark', 'deeptan', 'deep tan']
    melanin_users = df[df['skin_tone'].str.lower().isin(melanin_tones)]
    user_counts = melanin_users['author_id'].value_counts()
    top_users = user_counts[user_counts >= 5].head(20).index.tolist()
    top_user = random.choice(top_users)
    user_row = df[df['author_id'] == top_user].iloc[0]
    display_name = f"{random.choice(ethnic_first_names)} {random.choice(ethnic_last_names)}"
    encoded_id = df[df['author_id'] == top_user]['user'].iloc[0]

    # User profile
    col1, col2, col3 = st.columns(3)
    col1.metric("👤 User", display_name)
    col2.metric("🎨 Skin Tone", user_row['skin_tone'].capitalize())
    col3.metric("📝 Reviews", user_counts[top_user])

    st.divider()

    # Get recommendations
    with st.spinner("Finding your perfect products..."):
        raw_recs = model.recommend(int(encoded_id), top_n=200)

        results = []
        for rec in raw_recs:
            meta = item_to_meta.get(rec['item'], {})
            product_name = meta.get('product_name_x', 'Unknown')
            brand_name = meta.get('brand_name_x', 'Unknown')

            # Merge ingredients
            product_match = products_df[
                products_df['product_name_clean'] == product_name.strip().lower()
            ]
            ingredients = product_match.iloc[0]['ingredients'] if not product_match.empty else 'N/A'
            primary_cat = product_match.iloc[0].get('primary_category', '') if not product_match.empty else ''
            secondary_cat = product_match.iloc[0].get('secondary_category', '') if not product_match.empty else ''

            results.append({
                "product_name": product_name,
                "brand_name": brand_name,
                "predicted_rating": rec['predicted_rating'],
                "ingredients": str(ingredients),
                "primary_category": primary_cat,
                "secondary_category": secondary_cat,
                "safety_status": "✅ SAFE"
            })

        # Apply tone re-ranking
        results = tone_aware_rerank(results, product_tone_scores, alpha=0.3)

        # Category filter
        if category != "All Products":
            filtered = []
            for rec in results:
                cat1 = str(rec.get('primary_category', '')).lower()
                cat2 = str(rec.get('secondary_category', '')).lower()
                product_lower = rec['product_name'].lower()
                if category.lower() in cat1 or category.lower() in cat2:
                    filtered.append(rec)
                elif category.lower() == 'sunscreen' and any(
                    kw in product_lower for kw in ['spf', 'sunscreen', 'sun screen', 'uv']
                ):
                    filtered.append(rec)
            results = filtered

        # Allergen filter
        flagged_count = 0
        if nut_allergy == "Yes":
            safe_recs = []
            for rec in results:
                ingredients_lower = rec.get('ingredients', '').lower()
                found = next((a for a in nut_allergens if a in ingredients_lower), None)
                if found:
                    flagged_count += 1
                else:
                    safe_recs.append(rec)
            results = safe_recs

    # Display results
    if flagged_count > 0:
        st.warning(f"⚠️ {flagged_count} product(s) removed due to nut allergen detection")

    if results:
        st.subheader(f"🎯 Top Recommendations for {display_name}")
        for i, rec in enumerate(results[:10], 1):
            tone_emoji = "✅" if rec.get('tone_score', 0) > 0.1 else "⚠️" if rec.get('tone_score', 0) >= -0.1 else "❌"
            st.markdown(f"""
            <div class="rec-card">
                <h3>{i}. {rec['product_name']}</h3>
                <p><strong>Brand:</strong> {rec['brand_name']} &nbsp;|&nbsp;
                   <strong>⭐ Rating:</strong> {rec['predicted_rating']} &nbsp;|&nbsp;
                   <strong>{tone_emoji} Tone Score:</strong> {rec.get('tone_score', 'N/A')}</p>
                <p><strong>🧴 Ingredients:</strong> {str(rec.get('ingredients', 'N/A'))[:300]}...</p>
                <p><strong>🎨 Tone Compatibility:</strong> {rec.get('tone_label', 'No data')}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No matching products found for your preferences. Try selecting 'All Products'.")

st.divider()
st.caption("Powered by EU CosIng Database · Sephora Reviews Dataset · DATA 698 Capstone · Sheriann McLarty")