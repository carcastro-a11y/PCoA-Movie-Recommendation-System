# 🎬 PCoA Movie Recommendation System
### Mood-Based Movie Rails via Principal Coordinates Analysis

> *"When someone opens the app at 10pm on a Monday, they want to feel something specific — not browse a category label."*

[![Live Dashboard](https://img.shields.io/badge/🎥%20Live%20Dashboard-Vercel-black?style=for-the-badge)](https://movie-dashboard-m0gc0gh4y-carcastro-a11ys-projects.vercel.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=flat-square)
![scipy](https://img.shields.io/badge/scipy-1.x-lightblue?style=flat-square)

---

## 📌 What This Project Does

This system replaces traditional genre-based content browsing with **mood-based movie rails** — curated shelves that reflect how a film *feels*, not just what category it belongs to.

Instead of labeling a movie as "Action" or "Thriller," this pipeline embeds 8,000 films into a 20-dimensional latent space using **Principal Coordinates Analysis (PCoA)** and a custom **IDF-weighted Manhattan distance** metric. Each dimension captures a distinct emotional or thematic axis (e.g., *The Dread Axis*, *The Romance Axis*, *The Hero's Journey Axis*). Rails are then constructed by selecting films that score at meaningful thresholds along these axes.

The result: a content discovery system where a film like *Space Jam* and *Guardians of the Galaxy* can share a rail ("Easygoing Adventure") even though they belong to entirely different genre boxes.

**➡️ Explore the interactive dashboard:** [movie-dashboard-m0gc0gh4y-carcastro-a11ys-projects.vercel.app](https://movie-dashboard-m0gc0gh4y-carcastro-a11ys-projects.vercel.app)

---

## 🗂️ Repository Structure

```
📦 pcoa-movie-rails
├── PCOA_Step_1.py                         # PCoA pipeline: distance → embedding → correlations
├── PCOA_Step_2_DIMENSION_TIERS.py         # Rail construction from dimension thresholds
├── dimension_feature_correlations_sorted.csv  # Spearman correlations per dimension
├── Rail_Creation.pdf                      # Full methodology presentation
└── README.md
```

> ⚠️ **The underlying film dataset is proprietary** and not included in this repository. It was produced by professional film critics who scored 8,000 films across 248 trait features. See [Data](#-the-data) for details.

---

## 🎯 The Problem with Genre Tags

| Traditional Genre Label | Mood Rail |
|---|---|
| Binary — one film, one box | Multi-rail — one film can live in many rails |
| Editorial and static | Data-driven and dynamic |
| Describes *what* a film is | Describes *how* a film feels |
| `Action`, `Drama`, `Thriller` | `Easygoing Adventure`, `Dark Psychological Tension`, `Pressure Building` |

A genre tag tells a viewer *what* they're watching. A mood rail tells them *how they'll feel* watching it. That distinction is the entire product.

---

## 🗄️ The Data

The dataset used in this project was created by professional film critics and **cannot be shared publicly** due to licensing restrictions.

| Property | Value |
|---|---|
| Films | 8,000 |
| Trait features | 248 |
| Scoring scale | 0 (absent) → 3 (strongly present) |
| Sparsity | ~76.4% zero-valued cells |

**Example traits:** `suspenseful_atmosphere`, `character_driven`, `set_in_japan`, `nonsensical_comedy`, `happy_ending`, `psychological_suspense`

Each film was evaluated by experienced critics who assigned a score of 0–3 for every trait. A score of **0** means the trait is essentially absent from the film; **3** means it is a dominant, defining characteristic. This ordinal structure is fundamental to the distance and weighting decisions described below.

---

## ⚙️ The Pipeline

### Step 1 — PCoA Embedding (`PCOA_Step_1.py`)

#### 1. IDF Weighting
Traits that appear in only a handful of films carry more discriminative signal than near-universal traits. We apply a tempered **Inverse Document Frequency (IDF)** weight to each trait before computing distances:

```
weight = ( log( (N+1) / (df+1) ) + 1 ) ^ 0.85
```

- `N` = total films (8,000)
- `df` = number of films where trait > 0
- Exponent `0.85` dampens the influence of extremely rare traits, preventing noise from dominating

**Weight range: 1.19× – 3.476×**

| Trait | Films with trait | Weight |
|---|---|---|
| `set_in_japan` | 285 | 3.476 — very rare, very informative |
| `character_driven` | 6,371 | 1.190 — near-universal, low weight |

> **Why not Z-score?** All 248 traits share the same 0–3 ordinal scale. Z-scoring would amplify near-zero-variance features and destroy the intensity signal carried by values 1, 2, and 3.

#### 2. Weighted Manhattan Distance
Every pair among the 8,000 films receives a distance score:

```
distance(A, B) = Σ | weighted_trait_A - weighted_trait_B |
```

Manhattan distance is chosen over Euclidean because it respects the ordinal nature of the data — it treats each unit step on the 0–3 scale equally rather than squaring differences, which would over-penalise large gaps.

- Raw distance range: **7.19 – 617.94**
- Normalised to **[0, 1]** before embedding

#### 3. Classical MDS / PCoA Decomposition
The normalised distance matrix is double-centred and decomposed via eigenvalue decomposition (via `scipy.linalg.eigh`):

```python
H = I - (1/n) * ones_matrix
B = -0.5 * H @ D² @ H
eigenvalues, eigenvectors = eigh(B)
```

Coordinates are extracted as:
```
coords = eigenvectors[:, :k] @ diag(sqrt(abs(eigenvalues[:k])))
```

**20 dimensions** were retained based on the scree plot elbow and the marginal variance contribution curve. Dimensions 21–30 combined contribute less than Dimension 1 alone (10.3%), making the 20D cut-off well-justified.

| Variance captured | Value |
|---|---|
| Total (20D) | **47.4%** of positive eigenvalue variance |
| Dim 1 alone | 10.3% |
| Dims 21–30 combined | ~6.1% |
| Positive eigenvalues | 671 of 8,000 (expected for L1 metric) |

#### 4. Spearman Feature–Dimension Correlations
To interpret each dimension, we compute **Spearman rank correlations** between each dimension's coordinate scores and each of the 248 original traits across all 8,000 films.

Spearman is used (not Pearson) because the trait data is ordinal. Pearson would treat the gap between 0 and 1 as mathematically identical to the gap between 2 and 3 — Spearman works on ranks, correctly handling the stepped ordinal structure.

Results are saved to `dimension_feature_correlations_sorted.csv`.

**Top correlations for Dimension 1:**

| Feature | Correlation |
|---|---|
| Suspenseful atmosphere | +0.807 |
| Somber atmosphere | +0.766 |
| Psychological Suspense | +0.736 |
| Happy Ending | −0.680 |
| Positive, Uplifting Emotional Tone | −0.675 |
| Feel-Good Narrative | −0.652 |

This bipolar structure is how Dimension 1 earns the name **"The Dread Axis"**: high scores → dark psychological tension; low scores → warmth and uplift.

---

### Step 2 — Rail Construction (`PCOA_Step_2_DIMENSION_TIERS.py`)

#### The 20 Named Dimensions

| Dim | Name | Positive End | Negative End |
|---|---|---|---|
| 1 | The Dread Axis | Dark Psychological Tension | Warmth & Uplift |
| 2 | The Hero's Journey Axis | Heroic Adventure & Action | Psychological Unease |
| 3 | The Soul Depth Axis | Introspective Character Study | Kinetic Plot Drive |
| 4 | The Sharp Wit Axis | Clever Crime & Wit | Supernatural Horror |
| 5 | The Auteur Axis | Visionary Originality & Fantasy | Devoted Family Stories |
| 6 | The Twist Axis | Surprise-Driven & Twisty Storytelling | Historical War Epic |
| 7 | The American Sci-Fi Axis | American Technological Thriller | Asian Cinema |
| 8 | The Romance Axis | Romantic Drama & Female-Led Stories | Laddish Comedy & Male Bonds |
| 9 | The Future Tech Axis | Science Fiction & Future Technology | European Historical Period |
| 10 | The European Detective Axis | Old-World Mystery & Intellectual Intrigue | Raw Romantic Drama |
| 11 | The Medieval Betrayal Axis | Fantasy Worlds & Moral Treachery | Music-Driven Sound Design |
| 12 | The Father Figure Axis | Paternal Drama & Family Bonds | Female Underdog Stories |
| 13 | The Parental Warmth Axis | Family Comedy & Devoted Parents | Coming-of-Age Youth |
| 14 | The Redemption Axis | Inner Darkness & Moral Redemption | Female Coming-of-Age |
| 15 | The American Spotlight Axis | American Arts & Entertainment Culture | British/European Grit & Mortality |
| 16 | The Classic Era Axis | Black-and-White Timelessness | Contemporary Artist Drama |
| 17 | The European Arts Axis | European High Arts & Feminist Narrative | American Supernatural Fantasy |
| 18 | The Moral Corruption Axis | Temptation, Manipulation & Greed | Contemporary Urban Life |
| 19 | The Perseverance Axis | Struggle, Setbacks & the Will to Overcome | Contemporary Intellectual Drama |
| 20 | The Small-Town Modernity Axis | Recent Rural & Village Stories | Late 20th Century Urban Era |

#### How Rails Are Built

Each rail is defined by selecting 1–2 dimensions and filtering films to a meaningful threshold along those axes. Films are scored by their **weighted distance** from the threshold and ranked; the top 50 are kept per rail.

```python
# Example: Rail 01 — Easygoing Adventure
criteria = [
    (1, "very_low", "below"),   # Low on The Dread Axis → Warmth & Uplift
    (3, "very_low", "below"),   # Low on The Soul Depth Axis → Kinetic Plot Drive
]
# Dimension weights: Dim1 = 0.66, Dim3 = 0.34
```

**Sample output — Rail 01: Easygoing Adventure (top films by match score):**

| Film | Match Score |
|---|---|
| Space Jam | 0.1416 |
| Sonic the Hedgehog 2 | 0.1019 |
| DC League of Super-Pets | 0.0910 |
| Jumanji: Welcome to the Jungle | 0.0758 |
| Guardians of the Galaxy | 0.0434 |

> Note: a film can appear in multiple rails simultaneously — this is a feature, not a bug. *Harry Potter and the Half-Blood Prince* correctly appears in both "Romance in Epic Worlds" and other rails that capture its psychological and adventure traits.

---

## 📊 Why PCoA Over Other Methods?

| Method | Accepts Custom Distance | Interpretable Axes | Verdict |
|---|---|---|---|
| **PCoA** | ✅ Yes | ✅ Yes — bipolar axes | ✅ **Selected** |
| PCA | ❌ Euclidean only | ✅ Yes | ❌ Rejected |
| Factor Analysis | ❌ No | ✅ Yes | ❌ Rejected — assumes latent causality |
| Clustering | ✅ Yes | ❌ No axes | ❌ Rejected — wanted custom rail slicing |

PCoA is the only method that simultaneously accepts a custom IDF-weighted distance metric **and** produces consistent, nameable bipolar axes suitable for rail construction.

---

## 🖥️ Live Dashboard

The interactive dashboard lets you explore all 20 dimensions, adjust thresholds, and preview which films fall into each rail in real time.

**➡️ [Open the Dashboard](https://movie-dashboard-m0gc0gh4y-carcastro-a11ys-projects.vercel.app)**

Built with Vercel. The dashboard connects to the pre-computed PCoA coordinates and dimension correlation outputs from this pipeline.

---

## 🚀 Running the Pipeline

> You will need your own film trait dataset in the same format described above (8,000 × 248 ordinal matrix). The proprietary dataset used in this project is not redistributable.

**Install dependencies:**
```bash
pip install pandas numpy scikit-learn scipy matplotlib
```

**Run Step 1 — PCoA embedding & correlations:**
```bash
python PCOA_Step_1.py
# Outputs: pcoa_coordinates_20D.csv, dimension_feature_correlations.csv,
#          pcoa_eigen_spectrum.csv, pcoa_scree_plot.png, vibe_dimensions_report.txt
```

**Run Step 2 — Rail construction:**
```bash
python PCOA_Step_2_DIMENSION_TIERS.py
# Outputs: rail definitions with top 50 films per rail, scored by weighted match
```

---

## 👥 Team

**Becker · Castro · Hammer · Reyes**

---

## 📄 License

This repository contains pipeline code only. The underlying film trait dataset is proprietary and owned by its respective rights holders. All code in this repository is available for educational and research purposes.
