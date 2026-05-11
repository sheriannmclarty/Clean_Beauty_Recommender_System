import pandas as pd

sensitivity_terms = [
    'eczema', 'rash', 'breakout', 'irritat', 'reaction',
    'burn', 'sting', 'flare', 'allergic', 'hives', 'sensitive skin'
]

def compute_sensitivity_score(review_text: str) -> float:
    if not isinstance(review_text, str):
        return 0.0
    text = review_text.lower()
    count = sum(1 for term in sensitivity_terms if term in text)
    return round(min(count / 3, 1.0), 3)
# === Tone signal keywords for melanin-rich skincare ===
negative_signals = [
    'white cast', 'gray cast', 'grey cast', 'ashy', 'ash',
    'flashback', 'flash back', 'too heavy', 'looks gray',
    'looks ashy', 'oxidize', 'oxidizes', 'oxidation',
    'too pale', 'chalky', 'cakey', 'doesn\'t work on dark',
    'not for dark skin', 'bad for melanin'
]

positive_signals = [
    'no white cast', 'no flashback', 'no ashy', 'no gray cast',
    'great for dark skin', 'works on deep', 'works for melanin',
    'perfect for brown skin', 'great for melanin rich',
    'good for darker', 'no cast', 'invisible on dark',
    'blends well on dark', 'no residue', 'absorbs well'
]

def compute_tone_score(review_text: str) -> float:
    if not isinstance(review_text, str):
        return 0.0
    text = review_text.lower()
    neg_count = sum(1 for signal in negative_signals if signal in text)
    pos_count = sum(1 for signal in positive_signals if signal in text)
    total = neg_count + pos_count
    if total == 0:
        return 0.0
    return round((pos_count - neg_count) / total, 3)

def get_product_tone_scores(df: pd.DataFrame,
                             category: str = None) -> pd.DataFrame:
    working_df = df.copy()

    # Filter by category if specified
    if category:
        working_df = working_df[
            working_df['secondary_category'].str.lower() == category.lower()
        ]

    print(f"✅ Reviews found: {len(working_df)}")

    # Compute tone score per review
    working_df = working_df.copy()
    working_df['tone_score'] = working_df['review_text'].apply(compute_tone_score)

    # Aggregate by product — keep only numeric columns
    product_tone = working_df.groupby('product_name_x').agg(
        avg_tone_score=('tone_score', 'mean'),
        review_count=('tone_score', 'count'),
        pos_signals=('tone_score', lambda x: (x > 0).sum()),
        neg_signals=('tone_score', lambda x: (x < 0).sum())
    ).reset_index()

    product_tone['avg_tone_score'] = product_tone['avg_tone_score'].round(3)
    product_tone = product_tone.sort_values('avg_tone_score', ascending=False)
    return product_tone

def tone_aware_rerank(recommendations: list, product_tone_scores: pd.DataFrame,
                       alpha: float = 0.3) -> list:
    tone_dict = product_tone_scores.set_index('product_name_x')['avg_tone_score'].to_dict()

    for rec in recommendations:
        try:
            product_name = rec.get('product_name', '')
            tone_score = float(tone_dict.get(product_name, 0.0))
            predicted_rating = float(rec.get('predicted_rating', 0.0))

            normalized_rating = (predicted_rating - 1.0) / 4.0
            combined_score = (1 - alpha) * normalized_rating + alpha * ((tone_score + 1) / 2)

            rec['tone_score'] = round(tone_score, 3)
            rec['combined_score'] = round(combined_score, 4)
            rec['tone_label'] = (
                "✅ Positive skin tone signal" if tone_score > 0.1
                else "❌ Negative skin tone signal" if tone_score < -0.1
                else "⚠️ Limited or mixed skin tone signal"
            )
        except Exception:
            rec['tone_score'] = 0.0
            rec['combined_score'] = 0.0
            rec['tone_label'] = "⚠️ Limited or mixed skin tone signal"

    recommendations.sort(key=lambda x: x.get('combined_score', 0.0), reverse=True)
    return recommendations

# === Demo ===
if __name__ == "__main__":
    df = pd.read_csv("data/filtered_skintone_reviews.csv", low_memory=False)
    df = df.dropna(subset=['author_id', 'product_name_x', 'review_text'])

    print("🔍 Computing tone scores for sunscreen products...")
    sunscreen_tone = get_product_tone_scores(df, category='Sunscreen')
    print(f"\n✅ Tone scores computed for {len(sunscreen_tone)} sunscreen products")

    print("\n🏆 Most Tone-Compatible Sunscreens for Melanin-Rich Skin:")
    print(sunscreen_tone[sunscreen_tone['review_count'] >= 3].head(10).to_string(index=False))

    print("\n❌ Sunscreens with Tone Concerns:")
    print(sunscreen_tone[sunscreen_tone['review_count'] >= 3].tail(10).to_string(index=False))

    print("\n🔍 Computing tone scores for moisturizers...")
    moisturizer_tone = get_product_tone_scores(df, category='Moisturizers')
    print(f"\n✅ Tone scores computed for {len(moisturizer_tone)} moisturizer products")
    print(moisturizer_tone[moisturizer_tone['review_count'] >= 3].head(10).to_string(index=False))
