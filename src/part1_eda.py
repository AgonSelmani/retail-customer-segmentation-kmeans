import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Global helper function for saving matplotlib figures
def save_figure(fig_name):
    save_dir = "outputs/"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{fig_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()




# 1. Import dataset and initial inspection
print("1. IMPORTING THE DATASET AND INSPECT DATA")
df = pd.read_csv(r"C:\Users\Agon\Desktop\FAKULTETI\MY_PROJECTS\CLUSTERING_PROJECT\data\raw\My_retail_data.csv")
print(f"Dimensions: {df.shape[0]} rows X {df.shape[1]} columns")
print("Column names:", list(df.columns))
print("\n\n")





# 2. Data structure overview
print("2. DATA STRUCTURE OVERVIEW")
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn info:")
df.info()
print("\n\n")




# 3. Missing data analysis
print("3. MISSING DATA ANALYSIS")
missing_counts = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    "Missing_Count": missing_counts,
    "Missing_Percent": missing_percent
}).sort_values("Missing_Percent", ascending=False)

print(missing_df[missing_df["Missing_Count"] > 0])

plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap="viridis")
plt.title("Missing Data Heatmap")
plt.tight_layout()
save_figure("eda_3_missing_data_heatmap")
print("\n\n")





# 4. Descriptive statistics
print("4. DESCRIPTIVE STATISTICS")
print(df.describe(include=[np.number]))
print("\nCategorical summary:")
print(df.describe(include=["object", "category"]))
print("\n\n")






# 5. Outlier detection (IQR)
print("5. OUTLIER DETECTION (IQR)")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
outlier_report = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    if len(outliers) > 0:
        outlier_report[col] = (len(outliers), len(outliers) / len(df) * 100)
        print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")
print("\n\n")







# 6. Numeric feature distributions
print("6. NUMERIC FEATURE DISTRIBUTIONS")
for col in numeric_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df[col].dropna(), bins=30, edgecolor="black", alpha=0.7)
    axes[0].axvline(df[col].mean(), color="red", linestyle="--",
                    label=f"Mean: {df[col].mean():.2f}")
    axes[0].axvline(df[col].median(), color="green", linestyle="--",
                    label=f"Median: {df[col].median():.2f}")
    axes[0].set_title(f"Distribution of {col}")
    axes[0].set_xlabel(col); axes[0].set_ylabel("Frequency"); axes[0].legend()

    axes[1].boxplot(df[col].dropna(), vert=True)
    axes[1].set_title(f"Boxplot of {col}")
    axes[1].set_ylabel(col)

    plt.tight_layout()
    save_figure(f"eda_6_distribution_{col}")
print("\n\n")







# 7. Feature-feature correlation analysis
print("7. CORRELATION ANALYSIS (FEATURE vs FEATURE)")
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm",
            center=0, fmt=".2f", square=True, linewidth=0.5)
plt.title("Correlation Matrix")
plt.tight_layout()
save_figure("eda_9_correlation_matrix")

high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr.append((corr_matrix.columns[i],
                              corr_matrix.columns[j],
                              corr_matrix.iloc[i, j]))

if high_corr:
    print("Highly correlated features (>0.7) — consider dropping one of each pair:")
    for a, b, v in high_corr:
        print(f"  {a} — {b}: {v:.2f}")
else:
    print("No highly correlated features found.")
print("\n\n")

# Design decision: Dropping 'Customer_seg' and 'Avg_UnitPrice' before clustering
# - 'Customer_seg' causes data leakage as it is derived from UnitPrice. Keeping it 
#   validates an existing human rule rather than discovering segments. Furthermore, 
#   it applies to transactions, not customers, losing meaning upon aggregation.
# - 'Avg_UnitPrice' is a perfect duplicate of 'UnitPrice'. Keeping both would 
#   double-count the same feature in distance calculations.
#
# Correct workflow applied: Aggregate to customer level (RFM) -> Scale features ->
# Apply K-Means to discover segments -> Assign new, algorithm-driven cluster labels.




# 8. Feature scaling check
print("8. FEATURE SCALING CHECK")
scale_summary = df[numeric_cols].agg(["min", "max", "mean", "std"]).T
scale_summary["range"] = scale_summary["max"] - scale_summary["min"]
print(scale_summary.sort_values("range", ascending=False))
print("\nFeatures with range > 100x the smallest range will dominate distances.")
print("\n\n")





# 9. Duplicate checks and categorical consistency
print("9. DUPLICATE CHECKS")
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")

print("\nUnique values for categorical columns:")
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    unique_vals = df[col].unique()
    print(f"\n{col}: {len(unique_vals)} unique values")
    if len(unique_vals) < 20:
        print(f"  Values: {sorted(map(str, unique_vals))}")
print("\n\n")





# 10. Action items
print("10. ACTION ITEMS")

# 10.1 Redundant features (feature-feature correlation)
if high_corr:
    print("\n[UL] Drop one of each highly correlated pair (>0.7):")
    for a, b, v in high_corr:
        print(f"   → Drop one of: {a} or {b}   (r = {v:.2f})")
else:
    print("\n[UL] No redundant features to drop.")

# 10.2 Low-variance / near-constant features
low_var_features = []
for col in numeric_cols:
    mean = df[col].mean()
    std  = df[col].std()
    cv   = std / abs(mean) if mean != 0 else np.inf
    top_share = df[col].value_counts(normalize=True).iloc[0]

    if cv < 0.01 or top_share > 0.99:
        low_var_features.append((col, cv, top_share))

if low_var_features:
    print("\n[UL] Consider dropping low-variance features:")
    for col, cv, top in low_var_features:
        print(f"   → {col}   (CV={cv:.4f}, top-value share={top:.2%})")
else:
    print("\n[UL] All numeric features show meaningful variance.")








# 11. EDA findings summary
# Provides a checklist of data decisions to be applied during the preprocessing phase.

print("=" * 70)
print("EDA FINDINGS SUMMARY — actions to apply in PREPROCESSING")
print("=" * 70)

# 11.1 Columns to drop
print("\n[DROP] The following columns should be removed in preprocessing:")
print("  - Customer_seg     -> rule-based bin on UnitPrice (leakage)")
print("  - Avg_UnitPrice    -> perfect duplicate of UnitPrice (r=1.00)")
print("  - Season, Year, Month -> constant columns")
print("  - Country          -> near-constant (92% 'United Kingdom')")
print("  - Description      -> free text, not a feature")

# 11.2 Duplicates to remove
print("\n[DUPLICATES] From Section 9:")
print(f"  - {df.duplicated().sum()} exact duplicate rows -> drop in preprocessing")

# 11.3 Aggregation decision
print("\n[AGGREGATION] Current data is TRANSACTION-LEVEL.")
print("  Customer segmentation requires CUSTOMER-LEVEL data.")
print("  What we will do: group by CustomerID and compute behavioral features.")
print("  - Exclude Recency: Transactions span only a 5-day window (no meaningful variance).")
print("  - Exclude customers with 0 purchases (returns that cancel out).")
print("  - Drop 'total_quantity' (redundant with monetary) and 'frequency' (near-constant).")
print("  - Final features: 'monetary', 'num_products', 'avg_unit_price'.")

# 11.4 Outliers and Scaling
print("\n[OUTLIERS & SCALING] Handling skewness and distances:")
print("  - Outliers: Extreme outliers will be removed using 1.5x IQR before clustering")
print("    to prevent K-Means from isolating the tail into artificial clusters.")
print("  - Scaling: Use RobustScaler since max/median > 10 for key features,")
print("    making it resistant to remaining high-value VIP customers.")

# 11.5 Model plan
print("\n[MODELING] Next-stage plan:")
print("  1. Load customer-level scaled data.")
print("  2. Evaluate optimal k using elbow and silhouette scores.")
print("  3. Fit K-Means clustering.")
print("  4. Profile clusters using mean/median values to assign business names.")
print("  5. Visualize segments using pairwise scatter plots.")
print("=" * 70)
