import pandas as pd

# === Load CosIng Annex data ===
annex2 = pd.read_csv(
    "data/cosing_annex_prohibited_v2.txt",
    skiprows=4,
    encoding='utf-8',
    on_bad_lines='skip'
)

annex3 = pd.read_csv(
    "data/cosing_annex3_restricted_v2.txt",
    skiprows=4,
    encoding='utf-8',
    on_bad_lines='skip'
)

# === Clean ingredient names ===
annex2['ingredient_clean'] = annex2['Chemical name / INN'].str.strip().str.lower()
annex3['ingredient_clean'] = annex3['Chemical name / INN'].str.strip().str.lower()

# === Build hazard lists ===
prohibited_list = set(annex2['ingredient_clean'].dropna())
restricted_list = set(annex3['ingredient_clean'].dropna())

print(f"✅ Prohibited ingredients loaded: {len(prohibited_list)}")
print(f"✅ Restricted ingredients loaded: {len(restricted_list)}")

# === INCI Filter Function ===
def check_ingredients(product_ingredients: str) -> dict:
    if not isinstance(product_ingredients, str):
        return {"status": "unknown", "flagged": []}

    ingredients = [i.strip().lower() for i in product_ingredients.split(',')]
    prohibited_found = []
    restricted_found = []

    for ingredient in ingredients:
        # Only match if ingredient is a standalone word match
        # not just a substring inside a longer chemical description
        ing_words = set(ingredient.split())

        for p in prohibited_list:
            p_words = set(p.split())
            # Must share at least 2 words OR be an exact match
            if ingredient == p or (len(ing_words) > 1 and ing_words.issubset(p_words)):
                prohibited_found.append(f"{ingredient} → {p}")
                break

        for r in restricted_list:
            r_words = set(r.split())
            if ingredient == r or (len(ing_words) > 1 and ing_words.issubset(r_words)):
                restricted_found.append(f"{ingredient} → {r}")
                break

    if prohibited_found:
        return {"status": "❌ BLOCKED", "reason": "Contains EU prohibited ingredient(s)", "flagged": prohibited_found}
    elif restricted_found:
        return {"status": "⚠️ WARNING", "reason": "Contains EU restricted ingredient(s)", "flagged": restricted_found}
    else:
        return {"status": "✅ SAFE", "reason": "No prohibited or restricted ingredients found", "flagged": []}
# === Demo Test ===
print("\n🔍 Running demo tests...")

test_products = [
    {
        "name": "Test Moisturizer",
        "ingredients": "water, glycerin, niacinamide, phenoxyethanol"
    },
    {
        "name": "Risky Hair Product",
        "ingredients": "water, lead di(acetate), glycerin"
    },
    {
        "name": "Restricted Product",
        "ingredients": "water, thioglycolic acid, shea butter"
    },
    {
        "name": "Hair Relaxer",
        "ingredients": "water, formaldehyde, glycerin, cetyl alcohol"
    }
]
for product in test_products:
    result = check_ingredients(product['ingredients'])
    print(f"\nProduct: {product['name']}")
    print(f"Status:  {result['status']}")
    print(f"Reason:  {result['reason']}")
    if result['flagged']:
        print(f"Flagged: {result['flagged']}")
    print("🔍 Running demo tests...")