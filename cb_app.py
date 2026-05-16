import os
import gdown

def download_data():
    os.makedirs("data", exist_ok=True)

    files = {
        "data/product_info.csv": "1IQaIGcywItlj1gyUIErNDtT3-yOR7b7r",
        "data/filtered_skintone_reviews.csv": "1ozZm2XdzowyWsArPtQPNKiVDJHGkldJM",
        "data/cosing_annex_prohibited_v2.txt": "1go-cVOZtJmc0ELd_r-axxZZiyrJY5hsu",
        "data/cosing_annex3_restricted_v2.txt": "1Pl3Mc397zp25a6iJ8tkruHkbTgRpKjt_",
    }

    for path, file_id in files.items():
        # Skip only if file exists AND is large enough to be real data
        if os.path.exists(path) and os.path.getsize(path) > 10000:
            continue
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)
download_data()
import streamlit as st
import pandas as pd
import random
from cb_tone_ranking import get_product_tone_scores, tone_aware_rerank

st.set_page_config(page_title="Skintone Beauty Recommender", page_icon="🌿", layout="wide")

st.markdown("""<style>
    html, body, [data-testid="stAppViewContainer"], .stApp { background-color: #F5EFE8 !important; color: #2B1B24 !important; }
    [data-testid="stSidebar"] { background-color: #6A173D !important; }
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] span { color: #FFFFFF !important; }
    [data-testid="stSidebar"] div[data-baseweb="select"] * { color: #2B1B24 !important; }
    [data-testid="stSidebar"] input { color: #2B1B24 !important; }
    [data-testid="stSidebar"] div[role="radiogroup"] label span { color: #FFFFFF !important; }
    h1 { color: #6A173D !important; font-weight: 800 !important; }
    h2, h3 { color: #934261 !important; font-weight: 700 !important; }
    .stButton > button { background-color: #6A173D !important; color: white !important; border-radius: 10px !important; padding: 0.65em 2em !important; font-weight: 700 !important; border: none !important; }
    .stButton > button:hover { background-color: #934261 !important; color: white !important; }
    .rec-card { background-color: #FFF8F3 !important; color: #2B1B24 !important; padding: 1.25rem 1.4rem; border-radius: 16px; border-left: 7px solid #6A173D; margin-bottom: 1.1rem; box-shadow: 0 4px 14px rgba(92,29,58,0.14); }
    .rec-card * { color: #2B1B24 !important; }
    .rec-card h3 { color: #2B1B24 !important; margin-top: 0; margin-bottom: 0.55rem; font-size: 1.25rem; }
    .rec-card-caution { background-color: #FFF5EE !important; color: #2B1B24 !important; padding: 1.25rem 1.4rem; border-radius: 16px; border-left: 7px solid #C36A22; margin-bottom: 1.1rem; box-shadow: 0 4px 14px rgba(195,106,34,0.14); }
    .rec-card-caution * { color: #2B1B24 !important; }
    .rec-card-caution h3 { color: #2B1B24 !important; margin-top: 0; margin-bottom: 0.55rem; font-size: 1.25rem; }
    .pill { display: inline-block; background-color: #EFE6DF; color: #2B1B24 !important; padding: 0.25rem 0.6rem; border-radius: 999px; font-weight: 700; margin-right: 0.5rem; margin-bottom: 0.35rem; font-size: 0.9rem; }
    summary { cursor: pointer; font-weight: 700; color: #2B1B24 !important; }
</style>""", unsafe_allow_html=True)

# === Helpers ===
def clean_ingredients_text(ingredients):
    if pd.isna(ingredients):
        return "Ingredient list not available."
    text = str(ingredients).replace("[","").replace("]","").replace("'","").replace('"',"").strip()
    return text if text else "Ingredient list not available."

def explain_tone_signal(tone_score, category=""):
    try:
        tone_score = float(tone_score)
    except (TypeError, ValueError):
        tone_score = 0
    is_complexion = category == "Foundation & Complexion"
    if tone_score > 0.1:
        return {"label": "Positive skin tone review signal", "emoji": "✅",
                "explanation": "Reviews include positive tone-related language such as good match, no cast, works well on deeper skin, or better complexion performance."}
    elif tone_score < -0.1:
        return {"label": "Negative skin tone review signal", "emoji": "❌",
                "explanation": "Reviews include concern terms such as white cast, ashiness, oxidation, flashback, too orange, gray finish, or undertone mismatch."}
    else:
        return {"label": "Limited skin tone review signal", "emoji": "⚠️",
                "explanation": ("There was not enough complexion-specific review language to confirm shade or undertone performance." if is_complexion
                               else "There was not enough strong skin tone-specific review language to classify this product clearly.")}

def skin_type_match_score(rec, skin_type):
    if skin_type == "Not sure / No preference":
        return 0
    product_text = " ".join([str(rec.get("product_name","")), str(rec.get("ingredients","")),
        str(rec.get("primary_category","")), str(rec.get("secondary_category","")),
        str(rec.get("tertiary_category",""))]).lower()
    skin_type_keywords = {
        "Dry": ["hydrating","hydration","moisture","moisturizing","cream","barrier","ceramide","squalane","hyaluronic","balm"],
        "Oily": ["oil-free","oil free","gel","lightweight","matte","pore","clarifying","non-comedogenic","shine","sebum"],
        "Combination": ["balancing","lightweight","hydrating","gel cream","barrier","pore","daily"],
        "Normal": ["daily","hydrating","moisturizer","gentle","cream"],
        "Sensitive": ["sensitive","gentle","calming","soothing","fragrance-free","fragrance free","barrier","ceramide","cica","colloidal oatmeal"]
    }
    return sum(1 for kw in skin_type_keywords.get(skin_type, []) if kw in product_text)

def filter_by_category(results, category):
    keywords = category_keywords.get(category, [])
    return [r for r in results if any(
        kw in f"{r.get('primary_category','')} {r.get('secondary_category','')} {r.get('tertiary_category','')} {r.get('product_name','')}".lower()
        for kw in keywords
    )]

def get_users_for_tone(df, tone_labels, min_reviews=2):
    tone_users = df[df["skin_tone"].fillna("").str.lower().str.strip().isin(tone_labels)]
    user_counts = tone_users["author_id"].value_counts()
    return user_counts[user_counts >= min_reviews].index.tolist(), user_counts

# === Name pools ===
ethnic_first_names = [
    'Aaliyah','Imani','Zuri','Keisha','Latoya','Ebony','Destiny','Jasmine','Tamara','Shayla',
    'Monique','Tiana','Amara','Chioma','Fatima','Adaeze','Ngozi','Nia','Zainab','Aisha',
    'Blessing','Chiamaka','Folake','Yetunde','Gabriela','Valentina','Marisol','Xiomara',
    'Catalina','Lucia','Priya','Ananya','Divya','Meena','Kavya','Shreya','Nisha',
    'Sheriann','Sharai','Abena','Solange','Kaydean','Marsha','Andrea','Charmaine',
    'Juliet','Melissa','Jaleesa','Simone','Jessica','Angie','Nadine','Camille',
    'Donna','Sabrina','Stacyann','Rochelle','Tanisha','Khadijah','Naomi',
    'Temi','Tosin','Esi','Akosua','Sade','Kemi','Funmi','Yewande','Nkechi'
]
ethnic_last_names = [
    'Washington','Jackson','Williams','Thompson','Robinson','Okafor','Adeyemi','Nwosu',
    'Mensah','Diallo','Traore','Rodriguez','Martinez','Flores','Rivera','Reyes',
    'Patel','Sharma','Krishnan','Iyer','Nair','Smith','McLarty','Jean-Pierre',
    'Green','Lopez','Johnson','West','McIntosh','Roper','Gonzalez','Bailey',
    'James','Campbell','Hinds','Montgomery','Henry','Edwards','King','Clarke',
    'Reid','Grant','Francis','Bennett','Gordon','Boateng','Adebayo','Okonkwo',
    'Eze','Owusu','Santiago','Torres','Morales','Cruz','Persaud','Singh'
]

nut_allergens = [
    'shea butter','argan oil','almond oil','sweet almond oil','macadamia oil',
    'walnut oil','hazelnut oil','cashew','brazil nut','pecan','pistachio',
    'coconut oil','coconut acid','coconut water','cocos nucifera',
    'butyrospermum parkii','argania spinosa','prunus amygdalus','carthamus tinctorius'
]
sensitivity_terms = ['eczema','rash','breakout','irritat','reaction','burn','sting','flare','allergic','hives']

category_keywords = {
    "Moisturizers": ["moisturizer","moisturizers","cream","face cream","hydrating cream","daily moisturizer"],
    "Serums & Treatments": ["treatment","treatments","serum","retinol","acid","peel","exfoliant","brightening","dark spot","acne"],
    "Cleansers": ["cleanser","cleansing","wash","gel cleanser","foam","balm cleanser","oil cleanser"],
    "Sunscreen": ["sunscreen","spf","sun screen","uv","sun protection","mineral sunscreen"],
    "Foundation & Complexion": ["foundation","concealer","skin tint","tinted moisturizer","bb cream","cc cream","complexion","face makeup","setting powder","powder foundation","tint","mineral powder"]
}

skin_type_descriptions = {
    "Not sure / No preference": "No skin type preference will be used for ranking.",
    "Dry": "Prioritizes hydrating, cream, balm, and barrier-support products.",
    "Oily": "Prioritizes lightweight, gel, oil-free, non-greasy, and pore-friendly products.",
    "Combination": "Prioritizes balanced products that support both dry and oily areas.",
    "Normal": "Keeps broad product options without strong skin type filtering.",
    "Sensitive": "Prioritizes gentle, fragrance-free, calming, and barrier-support products."
}

skin_tone_options = {
    "Any Melanin-Rich Tone": {"labels": ["tan","deeptan","deep tan","dark","deep","rich","ebony"], "hex": "#8A5536", "description": "Uses any available melanin-rich user profile."},
    "Tan / Deep Tan": {"labels": ["tan","deeptan","deep tan"], "hex": "#B8794D", "description": "Warm tan to deep tan complexion range."},
    "Dark": {"labels": ["dark"], "hex": "#8A5536", "description": "Dark brown complexion range."},
    "Deep": {"labels": ["deep"], "hex": "#6A3F2A", "description": "Deep brown complexion range."},
    "Rich": {"labels": ["rich"], "hex": "#4A2A1C", "description": "Rich deep complexion range."},
    "Ebony": {"labels": ["ebony"], "hex": "#2B1711", "description": "Ebony / deepest complexion range."}
}

shade_profiles = {"Tan / Deep Tan": ["tan","deeptan","deep tan"], "Dark": ["dark"], "Deep": ["deep"], "Rich": ["rich"], "Ebony": ["ebony"]}
demo_categories = ["Moisturizers","Serums & Treatments","Cleansers","Sunscreen","Foundation & Complexion"]

# === Load data ===
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(
            "data/filtered_skintone_reviews.csv",
            low_memory=False,
            encoding="utf-8",
            engine="python",
            on_bad_lines="skip"
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            "data/filtered_skintone_reviews.csv",
            low_memory=False,
            encoding="cp1252",
            engine="python",
            on_bad_lines="skip"
        )

    df = df.dropna(subset=["author_id", "product_name_x", "rating_x"])
    df["author_id"] = df["author_id"].astype(str)
    df["rating_x"] = pd.to_numeric(df["rating_x"], errors="coerce")
    df = df.dropna(subset=["rating_x"])

    df["user"] = df["author_id"].astype("category").cat.codes
    df["item"] = df["product_name_x"].astype("category").cat.codes

    return df

@st.cache_data
def load_products():
    products_df = pd.read_csv("data/product_info.csv", low_memory=False)
    products_df['product_name_clean'] = products_df['product_name'].str.strip().str.lower()
    return products_df

@st.cache_data
def load_tone_scores():
    try:
        tone_df = pd.read_csv(
            "data/filtered_skintone_reviews.csv",
            low_memory=False,
            encoding="utf-8",
            engine="python",
            on_bad_lines="skip"
        )
    except UnicodeDecodeError:
        tone_df = pd.read_csv(
            "data/filtered_skintone_reviews.csv",
            low_memory=False,
            encoding="cp1252",
            engine="python",
            on_bad_lines="skip"
        )

    tone_df = tone_df.dropna(subset=["author_id", "product_name_x", "review_text"])
    return get_product_tone_scores(tone_df)

@st.cache_data
def load_cosing():
    annex2 = pd.read_csv("data/cosing_annex_prohibited_v2.txt", skiprows=4, encoding='utf-8', on_bad_lines='skip')
    annex3 = pd.read_csv("data/cosing_annex3_restricted_v2.txt", skiprows=4, encoding='utf-8', on_bad_lines='skip')
    annex2['ingredient_clean'] = annex2['Chemical name / INN'].str.strip().str.lower()
    annex3['ingredient_clean'] = annex3['Chemical name / INN'].str.strip().str.lower()
    return set(annex2['ingredient_clean'].dropna()), set(annex3['ingredient_clean'].dropna())

@st.cache_data
def compute_product_sensitivity(df):
    return df.groupby('product_name_x')['review_text'].apply(
        lambda reviews: sum(any(term in str(r).lower() for term in sensitivity_terms) for r in reviews) / max(len(reviews), 1)
    ).to_dict()

class BiasAdjustedRecommender:
    def __init__(self, user_biases, item_biases, global_avg, ratings_df):
        self.user_biases = user_biases
        self.item_biases = item_biases
        self.global_avg = global_avg
        self.ratings_df = ratings_df

    def predict(self, user, item):
        raw = self.global_avg + self.user_biases.get(user,0) + self.item_biases.get(item,0)
        return round(min(max(raw, 1.0), 5.0), 2)

    def recommend(self, user_id, top_n=200):
        user_rated = self.ratings_df[self.ratings_df['user'] == user_id]['item'].tolist()
        unseen_items = set(self.ratings_df['item']) - set(user_rated)
        item_review_counts = self.ratings_df['item'].value_counts().to_dict()
        predictions = []
        for item in unseen_items:
            base_score = self.predict(user_id, item)
            popularity_factor = round(min(item_review_counts.get(item,1) / 5000, 0.02), 4)
            final_score = round(min(base_score + popularity_factor, 5.0), 2)
            predictions.append((item, final_score))
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [{"item": int(item), "predicted_rating": round(score,2)} for item, score in predictions[:top_n]]

@st.cache_resource
def build_model():
    df = load_data()
    global_avg = df['rating_x'].mean()
    user_bias = (df.groupby('user')['rating_x'].mean() - global_avg).to_dict()
    item_bias = (df.groupby('item')['rating_x'].mean() - global_avg).to_dict()
    item_to_meta = df.drop_duplicates(subset='item').set_index('item')[['product_name_x','brand_name_x']].to_dict(orient='index')
    return BiasAdjustedRecommender(user_bias, item_bias, global_avg, df), item_to_meta

# === Load ===
with st.spinner("Loading data and models..."):
    df = load_data()
    products_df = load_products()
    product_tone_scores = load_tone_scores()
    prohibited_list, restricted_list = load_cosing()
    product_sensitivity_scores = compute_product_sensitivity(df)
    model, item_to_meta = build_model()

# === Sidebar ===
st.sidebar.title("🎯 Your Preferences")
st.sidebar.markdown("---")

category = st.sidebar.selectbox("What are you looking for?",
    ["All Products","Moisturizers","Serums & Treatments","Cleansers","Sunscreen","Foundation & Complexion"])

st.sidebar.markdown("### 🎨 Choose Skin Tone Profile")
selected_tone_group = st.sidebar.radio("Select a skin tone range:", list(skin_tone_options.keys()), index=0)
selected_tone_info = skin_tone_options[selected_tone_group]
selected_hex = selected_tone_info["hex"]

st.sidebar.markdown(f"""
<div style="background:rgba(255,255,255,0.10);padding:14px;border-radius:16px;border:1px solid rgba(255,255,255,0.35);margin-top:10px;margin-bottom:12px;">
    <div style="display:flex;justify-content:center;margin-bottom:12px;">
        <div style="background:{selected_hex};width:76px;height:76px;border-radius:50%;border:4px solid #FFFFFF;box-shadow:inset 0 4px 10px rgba(255,255,255,0.12),inset 0 -6px 12px rgba(0,0,0,0.25),0 4px 12px rgba(0,0,0,0.35);"></div>
    </div>
    <div style="text-align:center;color:white;"><strong>{selected_tone_group}</strong><br><span style="font-size:0.88rem;">{selected_tone_info["description"]}</span></div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="font-size:0.9rem;font-weight:700;margin-top:8px;margin-bottom:8px;color:white;">Skin tone palette</div>
<div style="display:flex;gap:8px;align-items:center;margin-bottom:18px;">
    <div title="Tan" style="background:#B8794D;width:34px;height:34px;border-radius:50%;border:2px solid white;box-shadow:0 2px 6px rgba(0,0,0,.35);"></div>
    <div title="Dark" style="background:#8A5536;width:34px;height:34px;border-radius:50%;border:2px solid white;box-shadow:0 2px 6px rgba(0,0,0,.35);"></div>
    <div title="Deep" style="background:#6A3F2A;width:34px;height:34px;border-radius:50%;border:2px solid white;box-shadow:0 2px 6px rgba(0,0,0,.35);"></div>
    <div title="Rich" style="background:#4A2A1C;width:34px;height:34px;border-radius:50%;border:2px solid white;box-shadow:0 2px 6px rgba(0,0,0,.35);"></div>
    <div title="Ebony" style="background:#2B1711;width:34px;height:34px;border-radius:50%;border:2px solid white;box-shadow:0 2px 6px rgba(0,0,0,.35);"></div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
skin_type = st.sidebar.selectbox("What is your skin type?",
    ["Not sure / No preference","Dry","Oily","Combination","Normal","Sensitive"])
st.sidebar.caption(skin_type_descriptions[skin_type])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Ingredient Concerns")
nut_allergy = st.sidebar.radio("Do you have a nut allergy?", ["No","Yes"])
st.sidebar.markdown("---")
sensitive_skin = st.sidebar.radio("Do you have sensitive skin or eczema?", ["No","Yes — filter irritants"])
st.sidebar.markdown("---")
demo_mode = st.sidebar.checkbox("Use stable demo user", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**🛡️ Safety Standards**")
st.sidebar.markdown(f"✅ {len(prohibited_list):,} prohibited ingredients blocked")
st.sidebar.markdown(f"⚠️ {len(restricted_list):,} restricted ingredients flagged")
st.sidebar.markdown("*Powered by EU CosIng Database*")

# === Header ===
st.title("🌿 Skintone Beauty Recommender")
st.markdown("### Clean beauty recommendations built around safety, skin type, and skin tone.")
st.markdown("Choose your product category, skin tone profile, skin type, and ingredient concerns. "
            "Skintone screens products first, then ranks recommendations using review-based compatibility signals.")
st.divider()

st.markdown(f"""
<div style="background:#FFF8F3;border-left:7px solid #6A173D;padding:1rem 1.2rem;border-radius:16px;margin-bottom:1rem;box-shadow:0 3px 12px rgba(92,29,58,0.12);display:flex;align-items:center;gap:18px;">
    <div style="background:{selected_hex};width:58px;height:58px;border-radius:50%;border:3px solid #FFFFFF;box-shadow:inset 0 4px 8px rgba(255,255,255,0.12),inset 0 -5px 10px rgba(0,0,0,0.25),0 3px 8px rgba(0,0,0,0.20);flex-shrink:0;"></div>
    <div>
        <div style="color:#6A173D;font-weight:800;font-size:1rem;">Selected Skin Tone Profile</div>
        <div style="color:#2B1B24;font-size:1.05rem;">{selected_tone_group}</div>
        <div style="color:#6B4A59;font-size:0.9rem;">{selected_tone_info["description"]}</div>
    </div>
</div>
""", unsafe_allow_html=True)

if st.button("✨ Get My Recommendations"):

    selected_skin_tones = selected_tone_info["labels"]
    tone_filtered_users = df[df["skin_tone"].fillna("").str.lower().str.strip().isin(selected_skin_tones)]
    user_counts = tone_filtered_users["author_id"].value_counts()
    top_users = user_counts[user_counts >= 5].head(20).index.tolist()
    actual_tone_group_used = selected_tone_group

    if not top_users:
        top_users = user_counts[user_counts >= 2].head(20).index.tolist()
        actual_tone_group_used = f"{selected_tone_group} (limited review profile)"

    if not top_users:
        st.warning(f"Not enough review profiles found for {selected_tone_group}. Showing Any Melanin-Rich Tone instead.")
        fallback_tones = skin_tone_options["Any Melanin-Rich Tone"]["labels"]
        tone_filtered_users = df[df["skin_tone"].fillna("").str.lower().str.strip().isin(fallback_tones)]
        user_counts = tone_filtered_users["author_id"].value_counts()
        top_users = user_counts[user_counts >= 5].head(20).index.tolist()
        actual_tone_group_used = "Any Melanin-Rich Tone (fallback)"

    if not top_users:
        st.error("No eligible user profiles found.")
        st.stop()

    top_user = top_users[0] if demo_mode else random.choice(top_users)
    user_row = df[df["author_id"] == top_user].iloc[0]
    display_name = f"{random.choice(ethnic_first_names)} {random.choice(ethnic_last_names)}"
    encoded_id = df[df["author_id"] == top_user]["user"].iloc[0]
    skin_type_display = "No skin type preference" if skin_type == "Not sure / No preference" else skin_type

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:1rem 0 1.5rem 0;">
        <div style="background:#FFF8F3;padding:14px;border-radius:14px;"><div style="color:#6A173D;font-weight:700;">👤 User</div><div style="font-size:1.2rem;color:#2B1B24;">{display_name}</div></div>
        <div style="background:#FFF8F3;padding:14px;border-radius:14px;"><div style="color:#6A173D;font-weight:700;">🎨 Skin Tone</div><div style="font-size:1.35rem;color:#2B1B24;">{str(user_row["skin_tone"]).capitalize()}</div></div>
        <div style="background:#FFF8F3;padding:14px;border-radius:14px;"><div style="color:#6A173D;font-weight:700;">💧 Skin Type</div><div style="font-size:1.05rem;color:#2B1B24;">{skin_type_display}</div></div>
        <div style="background:#FFF8F3;padding:14px;border-radius:14px;"><div style="color:#6A173D;font-weight:700;">📝 Reviews</div><div style="font-size:1.35rem;color:#2B1B24;">{user_counts[top_user]}</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    with st.spinner("Finding your perfect products..."):
        raw_recs = model.recommend(int(encoded_id), top_n=200)

        results = []
        for rec in raw_recs:
            meta = item_to_meta.get(rec['item'], {})
            product_name = meta.get('product_name_x', 'Unknown')
            brand_name = meta.get('brand_name_x', 'Unknown')
            product_match = products_df[products_df['product_name_clean'] == product_name.strip().lower()]
            ingredients = clean_ingredients_text(product_match.iloc[0]['ingredients'] if not product_match.empty else None)
            primary_cat = str(product_match.iloc[0].get('primary_category','')) if not product_match.empty else ''
            secondary_cat = str(product_match.iloc[0].get('secondary_category','')) if not product_match.empty else ''
            tertiary_cat = str(product_match.iloc[0].get('tertiary_category','')) if not product_match.empty else ''
            results.append({
                "product_name": product_name, "brand_name": brand_name,
                "predicted_rating": rec['predicted_rating'], "ingredients": ingredients,
                "primary_category": primary_cat, "secondary_category": secondary_cat, "tertiary_category": tertiary_cat,
            })

        results = tone_aware_rerank(results, product_tone_scores, alpha=0.3)

        for rec in results:
            rec["skin_type_score"] = skin_type_match_score(rec, skin_type)
        if skin_type != "Not sure / No preference":
            results = sorted(results, key=lambda x: (x.get("skin_type_score",0), x.get("tone_score",0), x.get("predicted_rating",0)), reverse=True)

        if category != "All Products":
            results = filter_by_category(results, category)

        flagged_count = 0
        if nut_allergy == "Yes":
            safe_recs = []
            for rec in results:
                if next((a for a in nut_allergens if a in rec.get('ingredients','').lower()), None):
                    flagged_count += 1
                else:
                    safe_recs.append(rec)
            results = safe_recs

        irritant_removed = 0
        if sensitive_skin == "Yes — filter irritants":
            sensitive_safe = []
            for rec in results:
                if product_sensitivity_scores.get(rec['product_name'], 0) > 0.15:
                    irritant_removed += 1
                else:
                    sensitive_safe.append(rec)
            results = sensitive_safe

    if flagged_count > 0:
        st.warning(f"⚠️ {flagged_count} product(s) removed before ranking due to nut allergen detection.")
    if sensitive_skin == "Yes — filter irritants":
        if irritant_removed > 0:
            st.warning(f"🔬 {irritant_removed} product(s) removed before ranking due to sensitive skin irritant screening.")
        else:
            st.success("✅ No sensitive-skin irritants were detected in the displayed recommendation pool.")

    st.info("A high predicted rating does not automatically mean a product is the best skin tone match. "
            "Skintone separates ingredient safety from skin tone review signals so users can see both.")

    if results:
        best_matches = [r for r in results if float(r.get('tone_score',0) or 0) >= -0.1]
        use_caution = [r for r in results if float(r.get('tone_score',0) or 0) < -0.1]

        st.subheader(f"✨ Best Matches for {display_name}")
        st.caption("Products that passed ingredient screening and have neutral or positive skin tone review signals.")

        if best_matches:
            for i, rec in enumerate(best_matches[:10], 1):
                tone_info = explain_tone_signal(rec.get('tone_score',0), category)
                ingredients_text = rec.get("ingredients","N/A")
                ingredients_preview = ingredients_text[:180] + "..." if len(ingredients_text) > 180 else ingredients_text

                skin_type_line = "<!-- -->"
                if skin_type != "Not sure / No preference":
                    st_score = rec.get("skin_type_score", 0)
                    st_label = f"Matches {skin_type} skin keywords" if st_score > 0 else f"No strong {skin_type.lower()} skin type signal found"
                    skin_type_line = f'<p><strong>💧 Skin Type Match:</strong> {st_label}</p>'

                card_html = f"""
<div class="rec-card">
    <h3>{i}. {rec['product_name']}</h3>
    <p>
        <span class="pill">Brand: {rec['brand_name']}</span>
        <span class="pill">⭐ Predicted Rating: {rec['predicted_rating']} / 5.0</span>
        <span class="pill">{tone_info['emoji']} Skin Tone Review Signal</span>
    </p>
    <p><strong>🎨 Skin Tone Compatibility:</strong> {tone_info['emoji']} {tone_info['label']}</p>
    <p style="color:#6B4A59;font-size:0.9rem;line-height:1.45;">{tone_info['explanation']}</p>
    {skin_type_line}
    <p><strong>🛡️ Ingredient Safety Status:</strong> Passed filter-first screening</p>
    <p><strong>🧴 Ingredient Preview:</strong> {ingredients_preview}</p>
    <details>
        <summary>View full ingredient list</summary>
        <p>{ingredients_text}</p>
    </details>
</div>"""
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            st.info("No best match products found. Try 'All Products'.")

        if use_caution:
            st.subheader("⚠️ Use Caution: Skin Tone Signal Concerns")
            st.caption("These products passed ingredient safety screening but may have negative skin tone compatibility signals in review text.")
            for i, rec in enumerate(use_caution[:5], 1):
                tone_info = explain_tone_signal(rec.get('tone_score',0), category)
                ingredients_text = rec.get("ingredients","N/A")
                ingredients_preview = ingredients_text[:180] + "..." if len(ingredients_text) > 180 else ingredients_text

                skin_type_line = "<!-- -->"
                if skin_type != "Not sure / No preference":
                    st_score = rec.get("skin_type_score", 0)
                    st_label = f"Matches {skin_type} skin keywords" if st_score > 0 else f"No strong {skin_type.lower()} skin type signal found"
                    skin_type_line = f'<p><strong>💧 Skin Type Match:</strong> {st_label}</p>'

                caution_html = f"""
<div class="rec-card-caution">
    <h3>{i}. {rec['product_name']}</h3>
    <p>
        <span class="pill">Brand: {rec['brand_name']}</span>
        <span class="pill">⭐ Predicted Rating: {rec['predicted_rating']} / 5.0</span>
        <span class="pill">❌ Skin Tone Review Signal</span>
    </p>
    <p><strong>🎨 Skin Tone Compatibility:</strong> ❌ {tone_info['label']}</p>
    <p style="color:#6B4A59;font-size:0.9rem;line-height:1.45;">{tone_info['explanation']}</p>
    {skin_type_line}
    <p><strong>🛡️ Ingredient Safety Status:</strong> Passed ingredient safety screening</p>
    <p><em>This product passed ingredient safety screening but may have negative skin tone compatibility signals in review text.</em></p>
    <p><strong>🧴 Ingredient Preview:</strong> {ingredients_preview}</p>
    <details>
        <summary>View full ingredient list</summary>
        <p>{ingredients_text}</p>
    </details>
</div>"""
                st.markdown(caution_html, unsafe_allow_html=True)

    elif category == "Foundation & Complexion":
        st.warning("No foundation or complexion products were found in the current recommendation pool. "
                   "The active dataset is skincare-focused (42,715 skincare reviews). "
                   "Foundation tone-aware ranking is scoped as a Phase 3 extension requiring a dedicated complexion makeup review dataset.")
    else:
        st.info("No matching products found. Try selecting 'All Products'.")

    # Coverage table
    st.subheader("📊 Skin Tone Profile Coverage")
    st.caption("One sample recommendation per category across all shade profiles, showing Skintone supports multiple complexion ranges.")

    overview_rows = []
    for shade_name, tone_labels in shade_profiles.items():
        users, shade_user_counts = get_users_for_tone(df, tone_labels, min_reviews=2)
        if not users:
            row = {"Skin Tone": shade_name}
            for cat in demo_categories:
                row[cat] = "No profile available"
            overview_rows.append(row)
            continue
        selected_encoded_id = df[df["author_id"] == users[0]]["user"].iloc[0]
        shade_raw_recs = model.recommend(int(selected_encoded_id), top_n=200)
        shade_results = []
        for rec in shade_raw_recs:
            meta = item_to_meta.get(rec["item"], {})
            product_name = meta.get("product_name_x","Unknown")
            pm = products_df[products_df["product_name_clean"] == product_name.strip().lower()]
            shade_results.append({
                "product_name": product_name, "brand_name": meta.get("brand_name_x","Unknown"),
                "predicted_rating": rec["predicted_rating"],
                "ingredients": clean_ingredients_text(pm.iloc[0]["ingredients"] if not pm.empty else None),
                "primary_category": str(pm.iloc[0].get("primary_category","")) if not pm.empty else "",
                "secondary_category": str(pm.iloc[0].get("secondary_category","")) if not pm.empty else "",
                "tertiary_category": str(pm.iloc[0].get("tertiary_category","")) if not pm.empty else "",
            })
        shade_results = tone_aware_rerank(shade_results, product_tone_scores, alpha=0.3)
        row = {"Skin Tone": shade_name}
        for cat in demo_categories:
            cat_results = filter_by_category(shade_results, cat)
            if cat_results:
                top = cat_results[0]
                tone_info = explain_tone_signal(top.get("tone_score",0), cat)
                row[cat] = f"{top['product_name']} — {top['brand_name']} ({tone_info['label']})"
            else:
                row[cat] = "No match found"
        overview_rows.append(row)

    overview_df = pd.DataFrame(overview_rows)
    if overview_df.empty:
        st.warning("No profile coverage results were generated.")
    else:
        st.dataframe(overview_df, use_container_width=True)

    with st.expander("How Skintone works"):
        st.markdown("""
        **Skintone uses a filter-first hybrid recommendation approach:**
        1. **Safety screening:** Products are checked against EU CosIng prohibited and restricted ingredient lists.
        2. **Allergen filtering:** If selected, products with nut-derived ingredients are removed.
        3. **Sensitive skin screening:** If selected, products with high irritation signals in reviews are removed.
        4. **Skin type ranking:** Products are re-ranked based on keyword matches for the selected skin type.
        5. **Personalized ranking:** A bias-adjusted recommender estimates product ratings for the user profile.
        6. **Skin tone compatibility:** Review-based tone signals categorize products into Best Matches and Use Caution.

        The **Skin Tone Review Signal** is based on review language related to white cast, flashback, ashiness, oxidation, undertone mismatch, and positive match terms. It is not a medical or dermatological safety score.
        """)

st.divider()
st.caption("Powered by EU CosIng Database · Sephora Reviews Dataset · DATA 698 Capstone · Sheriann McLarty")