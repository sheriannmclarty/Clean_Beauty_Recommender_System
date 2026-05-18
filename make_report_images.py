"""
make_report_images.py

Run this once from your Clean_Beauty_Recommender_System project folder.

It automatically creates the image files needed by Skintone_Final_Report_Revised.qmd:

images/app_homepage.png
images/app_recommendations.png
images/pipeline_architecture.png
images/skin_tone_distribution.png
images/tone_scores_sunscreen.png
images/sensitivity_terms.png

The app images are clean report visuals/mockups based on the Streamlit prototype layout.
The chart images are generated from either your data files or fallback values.
"""

from pathlib import Path
import textwrap

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


IMAGES_DIR = Path("images")
DATA_DIR = Path("data")
IMAGES_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def safe_read_csv(path):
    """Read CSV with common encodings and fallback parsing."""
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, encoding="cp1252", low_memory=False)
        except Exception:
            try:
                return pd.read_csv(path, engine="python", on_bad_lines="skip")
            except Exception:
                return None


def wrap_text(text, width=34):
    return "\n".join(textwrap.wrap(str(text), width=width))


def save_and_close(fig, filename):
    out = IMAGES_DIR / filename
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out}")


# ------------------------------------------------------------
# 1. Skin Tone Distribution
# ------------------------------------------------------------

def make_skin_tone_distribution():
    df = safe_read_csv(DATA_DIR / "filtered_skintone_reviews.csv")

    if df is not None and "skin_tone" in df.columns:
        tone_counts = df["skin_tone"].dropna().astype(str).str.lower().str.strip().value_counts()
        keep = ["tan", "deeptan", "deep tan", "dark", "deep", "rich", "ebony"]
        tone_counts = tone_counts[tone_counts.index.isin(keep)]
        if tone_counts.empty:
            tone_counts = pd.Series({
                "tan": 14500, "deeptan": 8200, "dark": 7200, "deep": 6500, "rich": 4200, "ebony": 2100
            })
    else:
        tone_counts = pd.Series({
            "tan": 14500, "deeptan": 8200, "dark": 7200, "deep": 6500, "rich": 4200, "ebony": 2100
        })

    colors = {
        "tan": "#B8794D",
        "deeptan": "#A06030",
        "deep tan": "#A06030",
        "dark": "#8A5536",
        "deep": "#6A3F2A",
        "rich": "#4A2A1C",
        "ebony": "#2B1711",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#F5EFE8")
    ax.set_facecolor("#F5EFE8")

    bars = ax.bar(
        tone_counts.index,
        tone_counts.values,
        color=[colors.get(str(t).lower(), "#8A5536") for t in tone_counts.index],
        edgecolor="white"
    )

    ax.set_title("Review Distribution by Skin Tone Group", fontsize=15, fontweight="bold", color="#4A1025")
    ax.set_xlabel("Skin Tone Group", fontsize=11)
    ax.set_ylabel("Number of Reviews", fontsize=11)

    for bar, val in zip(bars, tone_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(val):,}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=25)
    plt.tight_layout()

    save_and_close(fig, "skin_tone_distribution.png")


# ------------------------------------------------------------
# 2. Pipeline Architecture
# ------------------------------------------------------------

def make_pipeline_architecture():
    steps = [
        ("User\nInput", "#8B3A5A"),
        ("CosIng\nSafety\nScreen", "#6A173D"),
        ("Allergen\nFilter", "#5A1A30"),
        ("Sensitive\nSkin\nFilter", "#4A1025"),
        ("Bias-Adjusted\nCF Ranking", "#3A0A1A"),
        ("Skin Tone\nReview\nSignal", "#2A0612"),
    ]

    fig, ax = plt.subplots(figsize=(13, 3.8))
    fig.patch.set_facecolor("#F5EFE8")
    ax.set_facecolor("#F5EFE8")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.axis("off")

    for i, (label, color) in enumerate(steps):
        x = 0.35 + i * 2.05
        box = patches.FancyBboxPatch(
            (x, 1.2), 1.55, 1.6,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor=color,
            edgecolor="white",
            linewidth=2
        )
        ax.add_patch(box)
        ax.text(
            x + 0.775,
            2.0,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold"
        )

        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + 1.86, 2.0),
                xytext=(x + 1.58, 2.0),
                arrowprops=dict(arrowstyle="->", color="#934261", lw=2)
            )

    ax.text(
        6.5,
        0.45,
        "Output: Best Matches + Use Caution recommendation cards with ingredient and skin tone explanations",
        ha="center",
        va="center",
        fontsize=11,
        color="#4A1025",
        style="italic"
    )

    ax.set_title("Skintone Filter-First Recommendation Pipeline", fontsize=16, fontweight="bold", color="#4A1025")

    save_and_close(fig, "pipeline_architecture.png")


# ------------------------------------------------------------
# 3. Sunscreen Tone Scores
# ------------------------------------------------------------

def make_tone_scores_sunscreen():
    products = [
        "PLAY 100% Mineral SPF 30",
        "Ultra Sun Protection SPF 50+",
        "Mineral Mattescreen SPF 40",
        "Mineral Sheerscreen SPF 30",
        "Clean Screen Mineral SPF 30",
        "N°41 Facial Sunscreen Mist",
        "Super Fluid UV Defense SPF 50+",
    ]

    scores = [-1.000, -0.738, -0.667, -0.529, -0.508, 0.000, 0.000]

    colors = [
        "#C0392B" if s < -0.1 else "#F39C12" if s == 0 else "#27AE60"
        for s in scores
    ]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    fig.patch.set_facecolor("#F5EFE8")
    ax.set_facecolor("#F5EFE8")

    bars = ax.barh(products, scores, color=colors, edgecolor="white", height=0.62)

    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(x=-0.1, color="#C0392B", linewidth=0.8, linestyle=":", alpha=0.6)

    ax.set_xlabel("Skin Tone Review Score", fontsize=11)
    ax.set_title(
        "Sunscreen Skin Tone Review Scores\nMineral/Zinc Oxide Products",
        fontsize=14,
        fontweight="bold",
        color="#4A1025"
    )

    for bar, score in zip(bars, scores):
        ax.text(
            score - 0.02 if score < 0 else score + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            ha="right" if score < 0 else "left",
            fontsize=9
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_and_close(fig, "tone_scores_sunscreen.png")


# ------------------------------------------------------------
# 4. Sensitivity Terms
# ------------------------------------------------------------

def make_sensitivity_terms():
    terms = ["sensitive", "irritat", "breakout", "sting", "burn", "reaction", "eczema", "flare", "rash"]
    counts = [4875, 2926, 2264, 2209, 1169, 629, 257, 174, 147]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#F5EFE8")
    ax.set_facecolor("#F5EFE8")

    bars = ax.bar(terms, counts, color="#6A173D", edgecolor="white")

    ax.set_title("Sensitivity-Related Terms in Melanin-Rich User Reviews", fontsize=14, fontweight="bold", color="#4A1025")
    ax.set_xlabel("Sensitivity Term", fontsize=11)
    ax.set_ylabel("Number of Reviews", fontsize=11)

    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=25)
    plt.tight_layout()

    save_and_close(fig, "sensitivity_terms.png")


# ------------------------------------------------------------
# 5. App Homepage Mockup
# ------------------------------------------------------------

def make_app_homepage():
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#F5EFE8")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Sidebar
    ax.add_patch(patches.Rectangle((0, 0), 3.1, 7, facecolor="#6A173D", edgecolor="none"))
    ax.text(0.25, 6.55, "🎯 Your Preferences", color="white", fontsize=14, fontweight="bold")
    sidebar_items = [
        "What are you looking for?",
        "All Products",
        "Choose Skin Tone Profile",
        "Any Melanin-Rich Tone",
        "Skin tone palette",
        "What is your skin type?",
        "Not sure / No preference",
        "Ingredient Concerns",
        "Nut allergy: No",
        "Sensitive skin: No",
        "Safety Standards",
        "1,740 prohibited blocked",
        "374 restricted flagged",
    ]
    y = 6.05
    for item in sidebar_items:
        fs = 9.2 if len(item) < 24 else 8.6
        weight = "bold" if item in ["Choose Skin Tone Profile", "Ingredient Concerns", "Safety Standards"] else "normal"
        ax.text(0.28, y, item, color="white", fontsize=fs, fontweight=weight)
        y -= 0.42

    # Swatches
    swatches = ["#B8794D", "#8A5536", "#6A3F2A", "#4A2A1C", "#2B1711"]
    for i, c in enumerate(swatches):
        ax.add_patch(patches.Circle((0.55 + i*0.42, 3.72), 0.15, facecolor=c, edgecolor="white", linewidth=1))

    # Main header
    ax.text(3.55, 6.45, "🌿 Skintone Beauty Recommender", color="#6A173D", fontsize=22, fontweight="bold")
    ax.text(3.55, 6.0, "Clean beauty recommendations built around safety, skin type, and skin tone.", color="#2B1B24", fontsize=12)
    ax.text(3.55, 5.68, "Choose your product category, skin tone profile, skin type, and ingredient concerns.", color="#2B1B24", fontsize=10)

    # Selected card
    card = patches.FancyBboxPatch((3.55, 4.45), 8.9, 0.95, boxstyle="round,pad=0.03,rounding_size=0.12",
                                  facecolor="#FFF8F3", edgecolor="#6A173D", linewidth=2)
    ax.add_patch(card)
    ax.add_patch(patches.Circle((4.05, 4.93), 0.28, facecolor="#8A5536", edgecolor="white", linewidth=2))
    ax.text(4.55, 5.08, "Selected Skin Tone Profile", color="#6A173D", fontsize=11, fontweight="bold")
    ax.text(4.55, 4.78, "Any Melanin-Rich Tone", color="#2B1B24", fontsize=12)
    ax.text(4.55, 4.55, "Uses any available melanin-rich user profile.", color="#6B4A59", fontsize=9)

    # Button
    button = patches.FancyBboxPatch((3.55, 3.75), 2.4, 0.45, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    facecolor="#6A173D", edgecolor="none")
    ax.add_patch(button)
    ax.text(4.75, 3.98, "✨ Get My Recommendations", ha="center", va="center", color="white", fontsize=10, fontweight="bold")

    # Note box
    note = patches.FancyBboxPatch((3.55, 2.65), 8.9, 0.78, boxstyle="round,pad=0.04,rounding_size=0.12",
                                  facecolor="#FFFFFF", edgecolor="#D4A5B5")
    ax.add_patch(note)
    ax.text(3.85, 3.17, "Prototype purpose", color="#6A173D", fontsize=11, fontweight="bold")
    ax.text(3.85, 2.87, "Filter first, then personalize with review-based skin tone compatibility signals.", color="#2B1B24", fontsize=10)

    # Footer
    ax.text(3.55, 0.5, "Powered by EU CosIng Database · Sephora Reviews Dataset · DATA 698 Capstone · Sheriann McLarty",
            color="#6B4A59", fontsize=9)

    save_and_close(fig, "app_homepage.png")


# ------------------------------------------------------------
# 6. App Recommendations Mockup
# ------------------------------------------------------------

def make_app_recommendations():
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor("#F5EFE8")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(0.5, 7.55, "✨ Best Matches for Demo Reviewer Profile", color="#6A173D", fontsize=20, fontweight="bold")
    ax.text(0.5, 7.18, "Products that passed ingredient screening and have neutral or positive skin tone review signals.",
            color="#2B1B24", fontsize=10)

    profile_labels = [
        ("👤 Demo Reviewer Profile", "Sheriann McLarty"),
        ("🎨 Skin Tone", "Deep"),
        ("💧 Skin Type", "Dry"),
        ("📝 Reviews", "13"),
    ]

    x = 0.5
    for title, value in profile_labels:
        ax.add_patch(patches.FancyBboxPatch((x, 6.35), 2.85, 0.55, boxstyle="round,pad=0.04,rounding_size=0.1",
                                            facecolor="#FFF8F3", edgecolor="#E4C8D3"))
        ax.text(x+0.15, 6.68, title, color="#6A173D", fontsize=8.5, fontweight="bold")
        ax.text(x+0.15, 6.47, value, color="#2B1B24", fontsize=10)
        x += 3.05

    recs = [
        ("1. Advanced Retinol Daily Repair Face Moisturizer SPF 30", "StriVectin", "5.0 / 5.0", "Limited skin tone review signal"),
        ("2. Algae + Zinc Sea Kale Mineral Sunscreen Sérum SPF 30", "MARA", "5.0 / 5.0", "Limited skin tone review signal"),
        ("3. Alpha Beta Daily Moisturizer", "Dr. Dennis Gross Skincare", "5.0 / 5.0", "Positive skin tone review signal"),
    ]

    y = 5.65
    for title, brand, rating, signal in recs:
        ax.add_patch(patches.FancyBboxPatch((0.5, y-0.95), 12, 0.85, boxstyle="round,pad=0.04,rounding_size=0.12",
                                            facecolor="#FFF8F3", edgecolor="#6A173D", linewidth=1.5))
        ax.text(0.75, y-0.28, title, color="#2B1B24", fontsize=11, fontweight="bold")
        ax.text(0.75, y-0.55, f"Brand: {brand}    ⭐ Predicted Rating: {rating}    ⚠️ {signal}", color="#2B1B24", fontsize=9)
        ax.text(0.75, y-0.77, "🛡️ Ingredient Safety Status: Passed filter-first screening", color="#2B1B24", fontsize=8.5)
        y -= 1.05

    ax.text(0.5, 2.05, "⚠️ Use Caution: Skin Tone Signal Concerns", color="#934261", fontsize=16, fontweight="bold")
    ax.text(0.5, 1.72, "These products passed ingredient screening but may have negative skin tone compatibility signals.",
            color="#2B1B24", fontsize=9.5)

    ax.add_patch(patches.FancyBboxPatch((0.5, 0.62), 12, 0.85, boxstyle="round,pad=0.04,rounding_size=0.12",
                                        facecolor="#FFF5EE", edgecolor="#C36A22", linewidth=1.5))
    ax.text(0.75, 1.27, "1. Broad Spectrum SPF 50 Sunscreen Face Cream", color="#2B1B24", fontsize=11, fontweight="bold")
    ax.text(0.75, 1.0, "Brand: CLINIQUE    ⭐ Predicted Rating: 5.0 / 5.0    ❌ Negative skin tone review signal",
            color="#2B1B24", fontsize=9)
    ax.text(0.75, 0.78, "Possible concerns: white cast, flashback, ashiness, oxidation, or undertone mismatch.",
            color="#2B1B24", fontsize=8.5)

    save_and_close(fig, "app_recommendations.png")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    make_skin_tone_distribution()
    make_pipeline_architecture()
    make_tone_scores_sunscreen()
    make_sensitivity_terms()
    make_app_homepage()
    make_app_recommendations()
    print("\nAll report images created in the images/ folder.")
    print("Now run: quarto render Skintone_Final_Report_Revised.qmd")
