import os
import pandas as pd


DIMENSION_MEANINGS = {
    1: {"name": "The Dread Axis", "plus": "Dark Psychological Tension", "minus": "Warmth & Uplift"},
    2: {"name": "The Hero's Journey Axis", "plus": "Heroic Adventure & Action", "minus": "Psychological Unease"},
    3: {"name": "The Soul Depth Axis", "plus": "Introspective Character Study", "minus": "Kinetic Plot Drive"},
    4: {"name": "The Sharp Wit Axis", "plus": "Clever Crime & Wit", "minus": "Supernatural Horror"},
    5: {"name": "The Auteur Axis", "plus": "Visionary Originality & Fantasy", "minus": "Devoted Family Stories"},
    6: {"name": "The Twist Axis", "plus": "Surprise-Driven & Twisty Storytelling", "minus": "Historical War Epic"},
    7: {"name": "The American Sci-Fi Axis", "plus": "American Technological Thriller", "minus": "Asian Cinema"},
    8: {"name": "The Romance Axis", "plus": "Romantic Drama & Female-Led Stories", "minus": "Laddish Comedy & Male Bonds"},
    9: {"name": "The Future Tech Axis", "plus": "Science Fiction & Future Technology", "minus": "European Historical Period"},
    10: {"name": "The European Detective Axis", "plus": "Old-World Mystery & Intellectual Intrigue", "minus": "Raw Romantic Drama"},
    11: {"name": "The Medieval Betrayal Axis", "plus": "Fantasy Worlds & Moral Treachery", "minus": "Music-Driven Sound Design"},
    12: {"name": "The Father Figure Axis", "plus": "Paternal Drama & Family Bonds", "minus": "Female Underdog Stories"},
    13: {"name": "The Parental Warmth Axis", "plus": "Family Comedy & Devoted Parents", "minus": "Coming-of-Age Youth"},
    14: {"name": "The Redemption Axis", "plus": "Inner Darkness & Moral Redemption", "minus": "Female Coming-of-Age"},
    15: {"name": "The American Spotlight Axis", "plus": "American Arts & Entertainment Culture", "minus": "British/European Grit & Mortality"},
    16: {"name": "The Classic Era Axis", "plus": "Black-and-White Timelessness", "minus": "Contemporary Artist Drama"},
    17: {"name": "The European Arts Axis", "plus": "European High Arts & Feminist Narrative", "minus": "American Supernatural Fantasy"},
    18: {"name": "The Moral Corruption Axis", "plus": "Temptation, Manipulation & Greed", "minus": "Contemporary Urban Life"},
    19: {"name": "The Perseverance Axis", "plus": "Struggle, Setbacks & the Will to Overcome", "minus": "Contemporary Intellectual Drama"},
    20: {"name": "The Small-Town Modernity Axis", "plus": "Recent Rural & Village Stories", "minus": "Late 20th Century Urban Era"},
}


# Direction constraints from your notes:
# - Use positive end only for 2, 7, 9, 14, 18, 20
POSITIVE_ONLY_DIMS = {2, 7, 9, 14, 18, 20}


def criterion_text(dim_num, threshold_type, direction, threshold_value):
    info = DIMENSION_MEANINGS[dim_num]
    side = info["plus"] if direction == "above" else info["minus"]
    arrow = "↑" if direction == "above" else "↓"
    return (
        f"Dim{dim_num} ({info['name']}): {side} | "
        f"{threshold_type.replace('_', ' ').upper()} {arrow} ({direction} {threshold_value:.3f})"
    )


def validate_criteria(criteria):
    for dim_num, _, direction in criteria:
        if dim_num in POSITIVE_ONLY_DIMS and direction != "above":
            return False
    return True


def compute_match_score(row, criteria, percentiles):
    score = 0.0
    for dim_num, threshold_type, direction in criteria:
        value = row[f"PCoA_Dim{dim_num}"]
        threshold_value = percentiles[f"PCoA_Dim{dim_num}"][threshold_type]
        if direction == "above":
            score += max(0, value - threshold_value)
        else:
            score += max(0, threshold_value - value)
    return score


def trim_broad_rail_to_target(matching_df, criteria, percentiles, target_count=50):
    if len(matching_df) <= target_count:
        trimmed = matching_df.copy()
        trimmed["_match_score"] = trimmed.apply(lambda row: compute_match_score(row, criteria, percentiles), axis=1)
        return trimmed, 100.0, False

    scored = matching_df.copy()
    scored["_match_score"] = scored.apply(lambda row: compute_match_score(row, criteria, percentiles), axis=1)
    keep_fraction = target_count / len(scored)
    percentile_cutoff = max(0.0, 1.0 - keep_fraction)
    score_cutoff = scored["_match_score"].quantile(percentile_cutoff)

    trimmed = scored[scored["_match_score"] >= score_cutoff].copy()
    if len(trimmed) > target_count:
        trimmed = trimmed.nlargest(target_count, "_match_score")

    return trimmed, (1.0 - percentile_cutoff) * 100.0, True


def generate_dimension_tier_rails(
    metadata_file="Cleaned_data.csv",
    coords_file="pcoa_coordinates_20D.csv",
    output_file="FEELING_BASED_RAILS_1D_2D_3D.txt",
    summary_file="FEELING_BASED_RAILS_1D_2D_3D_SUMMARY.csv",
):
    print("🎬 Initializing 1D/2D/3D Dimension-Tier Rail Generator...")

    if not all(os.path.exists(f) for f in [metadata_file, coords_file]):
        print("❌ Error: Required files not found.")
        return

    df_movies = pd.read_csv(metadata_file)
    df_coords = pd.read_csv(coords_file)
    df = pd.merge(df_movies[["imdb_id", "movie_name"]], df_coords, on="imdb_id", how="inner")

    dim_cols = [f"PCoA_Dim{i}" for i in range(1, 21)]
    percentiles = {}
    for dim in dim_cols:
        percentiles[dim] = {
            "bottom_1pct": df[dim].quantile(0.01),
            "bottom_2pct": df[dim].quantile(0.02),
            "extreme_low": df[dim].quantile(0.05),
            "very_low": df[dim].quantile(0.10),
            "low": df[dim].quantile(0.25),
            "medium": df[dim].quantile(0.50),
            "high": df[dim].quantile(0.75),
            "very_high": df[dim].quantile(0.90),
            "extreme_high": df[dim].quantile(0.95),
            "top_2pct": df[dim].quantile(0.98),
            "top_1pct": df[dim].quantile(0.99),
        }

    # ---------------------------------------------------------------------
    # Rail sets to compare dimensionality effects
    # ---------------------------------------------------------------------
    tier_1d = [
        ("1D_Dread_Dark_Intensity", [(1, "top_1pct", "above")]),
        ("1D_Dread_Warm_Uplift", [(1, "bottom_1pct", "below")]),
        ("1D_Heroic_Action_Core", [(2, "top_1pct", "above")]),
        ("1D_Character_Study_Core", [(3, "top_1pct", "above")]),
        ("1D_Twisty_Storytelling_Core", [(6, "top_1pct", "above")]),
        ("1D_American_Tech_Thriller_Core", [(7, "top_1pct", "above")]),
        ("1D_Romantic_Drama_Core", [(8, "top_1pct", "above")]),
        ("1D_Future_Tech_Core", [(9, "top_1pct", "above")]),
        ("1D_European_Detective_Core", [(10, "top_1pct", "above")]),
        ("1D_Redemption_Core", [(14, "top_1pct", "above")]),
    ]

    tier_2d = [
        ("2D_Dark_Twisty_Mystery", [(1, "very_high", "above"), (6, "very_high", "above")]),
        ("2D_Dark_European_Detective", [(1, "very_high", "above"), (10, "very_high", "above")]),
        ("2D_Heroic_Future_Tech", [(2, "very_high", "above"), (9, "very_high", "above")]),
        ("2D_Heroic_American_SciFi", [(2, "very_high", "above"), (7, "very_high", "above")]),
        ("2D_Introspective_Romantic", [(3, "very_high", "above"), (8, "very_high", "above")]),
        ("2D_Auteur_Twisty", [(5, "very_high", "above"), (6, "very_high", "above")]),
        ("2D_Moral_Redemption", [(18, "very_high", "above"), (14, "very_high", "above")]),
        ("2D_Parental_Redemption", [(12, "very_high", "above"), (14, "very_high", "above")]),
        ("2D_European_Art_CrimeWit", [(17, "very_high", "above"), (4, "very_high", "above")]),
        ("2D_Perseverance_Detective", [(19, "very_high", "above"), (10, "very_high", "above")]),
    ]

    tier_3d = [
        ("3D_Dread_Twist_Detective", [(1, "high", "above"), (6, "very_high", "above"), (10, "high", "above")]),
        ("3D_Heroic_SciFi_Future", [(2, "very_high", "above"), (7, "high", "above"), (9, "high", "above")]),
        ("3D_Dark_Redemption_Perseverance", [(1, "high", "above"), (14, "very_high", "above"), (19, "high", "above")]),
        ("3D_Introspective_Romance_Redemption", [(3, "high", "above"), (8, "high", "above"), (14, "high", "above")]),
        ("3D_Auteur_CleverCrime_Twists", [(5, "high", "above"), (4, "high", "above"), (6, "high", "above")]),
        ("3D_Parental_Warmth_FatherArc", [(12, "high", "above"), (13, "high", "above"), (1, "very_low", "below")]),
        ("3D_European_Intrigue_Arts", [(10, "high", "above"), (17, "high", "above"), (3, "high", "above")]),
        ("3D_MoralCorruption_Noir", [(18, "very_high", "above"), (1, "high", "above"), (6, "high", "above")]),
        ("3D_SmallTown_Redemption", [(20, "high", "above"), (14, "high", "above"), (19, "high", "above")]),
        ("3D_Heroic_CrimeWit_Detective", [(2, "high", "above"), (4, "high", "above"), (10, "high", "above")]),
    ]

    tiered_configs = [
        ("1D", tier_1d, (35, 80)),
        ("2D", tier_2d, (30, 90)),
        ("3D", tier_3d, (20, 90)),
    ]

    print(f"📊 Loaded {len(df)} movies with updated coordinates")
    print("\n🛠 Building rails by dimension tier...")

    all_rails = []
    summary_rows = []

    for tier_name, configs, (min_size, max_size) in tiered_configs:
        print(f"\n--- {tier_name} Rails (size window: {min_size}-{max_size}) ---")

        for rail_name, criteria in configs:
            if not validate_criteria(criteria):
                print(f"  ⚠ {rail_name:<40} | invalid direction rule -> skipped")
                continue

            mask = pd.Series([True] * len(df))
            for dim_num, threshold_type, direction in criteria:
                dim_col = f"PCoA_Dim{dim_num}"
                threshold_val = percentiles[dim_col][threshold_type]
                if direction == "above":
                    mask &= df[dim_col] >= threshold_val
                else:
                    mask &= df[dim_col] <= threshold_val

            matching = df[mask].copy()
            raw_count = len(matching)
            trimmed_to_top_percentile = False
            kept_top_percent = 100.0

            force_trim = tier_name == "1D"
            if raw_count > max_size or force_trim:
                matching, kept_top_percent, trimmed_to_top_percentile = trim_broad_rail_to_target(
                    matching, criteria, percentiles, target_count=50
                )
            else:
                matching = matching.copy()
                matching["_match_score"] = matching.apply(lambda row: compute_match_score(row, criteria, percentiles), axis=1)

            count = len(matching)

            if min_size <= count <= max_size:
                all_rails.append(
                    {
                        "tier": tier_name,
                        "name": rail_name,
                        "raw_count": raw_count,
                        "count": count,
                        "trimmed": trimmed_to_top_percentile,
                        "kept_top_percent": kept_top_percent,
                        "criteria": criteria,
                        "rows": matching,
                    }
                )
                if trimmed_to_top_percentile:
                    print(
                        f"  ✓ {rail_name:<40} | {count:>4} movies "
                        f"(from {raw_count}, top {kept_top_percent:.1f}% by match score)"
                    )
                else:
                    print(f"  ✓ {rail_name:<40} | {count:>4} movies")
            elif raw_count > max_size:
                print(f"  ⚠ {rail_name:<40} | {count:>4} movies after trim | STILL BROAD")
            else:
                print(f"  ⚠ {rail_name:<40} | {count:>4} movies | TOO NARROW")

            summary_rows.append(
                {
                    "tier": tier_name,
                    "rail_name": rail_name,
                    "raw_movie_count": raw_count,
                    "movie_count": count,
                    "trimmed": trimmed_to_top_percentile,
                    "kept_top_percent": round(kept_top_percent, 2),
                    "status": "kept" if min_size <= count <= max_size else ("too_broad" if count > max_size else "too_narrow"),
                }
            )

    if not all_rails:
        print("\n❌ No rails satisfied the configured constraints.")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("🎬 DIMENSION-TIER FEELING-BASED MOVIE RAILS (1D / 2D / 3D)\n")
        f.write("=" * 120 + "\n")
        f.write("Built from updated tempered-IDF coordinates and your semantic dimension template\n")
        f.write("Includes rail sets with 1, 2, and 3 dimensions to compare specificity effects\n")
        f.write("=" * 120 + "\n\n")

        for tier_name in ["1D", "2D", "3D"]:
            tier_rails = [r for r in all_rails if r["tier"] == tier_name]
            tier_rails = sorted(tier_rails, key=lambda x: x["count"], reverse=True)

            f.write("\n" + "#" * 120 + "\n")
            f.write(f"{tier_name} RAILS (kept: {len(tier_rails)})\n")
            f.write("#" * 120 + "\n")

            for idx, rail in enumerate(tier_rails, 1):
                f.write("\n" + "=" * 120 + "\n")
                f.write(f"{tier_name} RAIL {idx}: {rail['name']}\n")
                f.write("=" * 120 + "\n")
                f.write(f"Total movies in this rail: {rail['count']}\n")
                f.write(f"Pre-trim size: {rail['raw_count']}\n")
                if rail["trimmed"]:
                    f.write(f"Top-percentile refinement applied: kept top {rail['kept_top_percent']:.1f}% by match score\n")

                f.write("\n📊 DIMENSION PROFILE:\n")
                for dim_num, threshold_type, direction in rail["criteria"]:
                    threshold_value = percentiles[f"PCoA_Dim{dim_num}"][threshold_type]
                    f.write(f"   • {criterion_text(dim_num, threshold_type, direction, threshold_value)}\n")

                f.write("\n📽️  MOVIES IN THIS RAIL:\n")
                f.write("-" * 120 + "\n")
                rail_dims = [dim_num for dim_num, _, _ in rail["criteria"]]
                dim_headers = [f"Dim{dim_num}" for dim_num in rail_dims]
                header = f"{'Movie Name':<60} | " + " | ".join([f"{h:>7}" for h in dim_headers])
                f.write(header + "\n")
                f.write("-" * 120 + "\n")

                rows_sorted = rail["rows"].sort_values(by="_match_score", ascending=False)
                for _, row in rows_sorted.iterrows():
                    dim_values = [row[f"PCoA_Dim{dim_num}"] for dim_num in rail_dims]
                    dim_text = " | ".join([f"{v:>7.3f}" for v in dim_values])
                    f.write(f"{row['movie_name'][:59]:<60} | {dim_text}\n")

                f.write("\n")

        f.write("\n" + "=" * 120 + "\n")
        f.write("NOTES APPLIED FROM DIMENSION GUIDANCE\n")
        f.write("=" * 120 + "\n")
        f.write("• Positive-end only enforced for dims: 2, 7, 9, 14, 18, 20\n")
        f.write("• Dim16 treated as secondary only (not used as a primary 1D rail)\n")
        f.write("• 1D rails use top/bottom 1% thresholds before match-score ranking\n")
        f.write("• Broad rails are refined to ~50 movies by top-percentile match score\n")

    pd.DataFrame(summary_rows).to_csv(summary_file, index=False)

    kept_counts = pd.DataFrame(all_rails).groupby("tier")["name"].count().to_dict()
    print("\n✅ Success!")
    print(f"📄 Output rails: {output_file}")
    print(f"📄 Summary CSV: {summary_file}")
    print(
        "📊 Kept rails by tier: "
        + ", ".join([f"{tier}={kept_counts.get(tier, 0)}" for tier in ["1D", "2D", "3D"]])
    )


if __name__ == "__main__":
    generate_dimension_tier_rails(
        metadata_file="Cleaned_data.csv",
        coords_file="pcoa_coordinates_20D.csv",
        output_file="FEELING_BASED_RAILS_1D_2D_3D.txt",
        summary_file="FEELING_BASED_RAILS_1D_2D_3D_SUMMARY.csv",
    )
