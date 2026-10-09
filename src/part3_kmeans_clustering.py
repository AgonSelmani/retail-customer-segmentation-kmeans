import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# 1. Load processed data
# Loads original data for interpreting clusters and scaled data for K-Means input.
print("1. Loading processed data")

customer_df = pd.read_csv('data/processed/customer_level.csv')
X_scaled_df = pd.read_csv('data/processed/customer_level_scaled.csv')

print(f"   customer_df : {customer_df.shape}")
print(f"   X_scaled_df : {X_scaled_df.shape}")

assert customer_df.shape[0] == X_scaled_df.shape[0], "Row counts do not match"
print("   Loaded successfully")
print("\n\n")


# 2. Initial elbow and silhouette analysis (before outlier removal)
# Evaluates raw clustering behavior. Outlier-heavy data often causes an artificial silhouette 
# spike at k=2 because K-Means attempts to isolate the extreme tail into its own cluster.
print("2. Initial elbow + silhouette (before outlier removal)")

K_RANGE = range(2, 11)
inertias_init = []
silhouettes_init = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled_df)
    inertias_init.append(km.inertia_)
    silhouettes_init.append(silhouette_score(X_scaled_df, labels))
    print(f"   k={k:2d}  inertia={km.inertia_:8.2f}   silhouette={silhouettes_init[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(K_RANGE), inertias_init, marker='o', linewidth=2)
axes[0].set_title('Elbow — BEFORE outlier removal')
axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia'); axes[0].grid(alpha=0.3)

axes[1].plot(list(K_RANGE), silhouettes_init, marker='o', linewidth=2, color='green')
axes[1].set_title('Silhouette — BEFORE outlier removal')
axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette'); axes[1].grid(alpha=0.3)

import os
os.makedirs('outputs', exist_ok=True)
plt.tight_layout()
plt.savefig('outputs/elbow_silhouette_before.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n\n")


# 3. Handle extreme outliers
# Removes extreme outliers that distort clustering and cause the artificial k=2 silhouette spike.
# Filters on both 'monetary' and 'avg_unit_price' using 1.5x IQR to ensure
# segmentation focuses on the typical customer population.
print("3. Handling extreme outliers (IQR 1.5x on monetary + avg_unit_price)")

def iqr_upper_bound(series, k=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return Q3 + k * IQR

# 3.1 Compute bounds
mon_ub = iqr_upper_bound(customer_df['monetary'])
aup_ub = iqr_upper_bound(customer_df['avg_unit_price'])

print(f"\n   monetary       upper bound: £{mon_ub:,.2f}")
print(f"   avg_unit_price upper bound: £{aup_ub:,.2f}")

# 3.2 Identify outliers in either feature
out_mask = (customer_df['monetary'] > mon_ub) | (customer_df['avg_unit_price'] > aup_ub)
n_outliers = out_mask.sum()

print(f"\n   Outliers to remove: {n_outliers}")
print(customer_df[out_mask][['CustomerID', 'monetary',
                             'num_products', 'avg_unit_price']])

# 3.3 Remove outliers from both dataframes while maintaining alignment
customer_df = customer_df[~out_mask].reset_index(drop=True)
X_scaled_df = X_scaled_df[~out_mask].reset_index(drop=True)

print(f"\n   Removed: {n_outliers} customer(s)")
print(f"   Remaining: {customer_df.shape[0]} customers")

# 3.4 Verify removal
assert customer_df.shape[0] == X_scaled_df.shape[0], "Row counts diverged"
assert (customer_df['monetary'] <= mon_ub).all(), "monetary outliers remain"
assert (customer_df['avg_unit_price'] <= aup_ub).all(), "avg_unit_price outliers remain"
print("   Both dataframes aligned, no outliers remain")
print("\n\n")


# 4. Elbow and silhouette analysis (after outlier removal)
# Re-runs diagnostics on cleaned data. The true optimal k should emerge
# where the elbow bends and the silhouette peaks in the meaningful range (k >= 3).
# The k=2 spike is ignored as a known artifact of outlier isolation.
print("4. Elbow + silhouette (after outlier removal)")

inertias = []
silhouettes = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled_df)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled_df, labels))
    print(f"   k={k:2d}  inertia={km.inertia_:8.2f}   silhouette={silhouettes[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(K_RANGE), inertias, marker='o', linewidth=2)
axes[0].set_title('Elbow — AFTER outlier removal')
axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia'); axes[0].grid(alpha=0.3)

axes[1].plot(list(K_RANGE), silhouettes, marker='o', linewidth=2, color='green')
axes[1].set_title('Silhouette — AFTER outlier removal')
axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette'); axes[1].grid(alpha=0.3)

import os
os.makedirs('outputs', exist_ok=True)
plt.tight_layout()
plt.savefig('outputs/elbow_silhouette_after.png', dpi=300, bbox_inches='tight')
plt.close()

# Ignore k=2 spike (known artifact); find best k in the meaningful range
k_list = list(K_RANGE)
sil_no_k2 = silhouettes[1:]   # skip k=2
best_k_clean = k_list[1:][int(np.argmax(sil_no_k2))]
print(f"\n   Best k in range [3,10]: {best_k_clean}  "
      f"(silhouette = {max(sil_no_k2):.4f})")
print("\n\n")


# 5. Fit K-Means and profile clusters
# Design decision: k=4 is chosen because the elbow bends at k=4 and the silhouette peaks
# there (within the meaningful k >= 3 range). While k=2 has a higher silhouette, it 
# trivially splits "high spenders vs everyone else". k=4 produces distinct, actionable 
# behavioral segments.
print("5. Fitting K-Means with k=4")

K = 4
km = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled_df)

print(f"\n   Inertia          : {km.inertia_:.2f}")
print(f"   Silhouette score : {silhouette_score(X_scaled_df, labels):.4f}")

# 5.1 Attach cluster labels to original-unit dataframe
customer_df['cluster'] = labels

print(f"\n   Cluster sizes:")
print(customer_df['cluster'].value_counts().sort_index())

# 5.2 Profile clusters using mean and median values for interpretability
print(f"\n   Cluster profiles (MEAN values per cluster):")
profile_mean = customer_df.groupby('cluster')[
    ['monetary', 'num_products', 'avg_unit_price']
].mean().round(2)
profile_mean['count'] = customer_df['cluster'].value_counts().sort_index()
print(profile_mean)

print(f"\n   Cluster profiles (MEDIAN values per cluster):")
profile_med = customer_df.groupby('cluster')[
    ['monetary', 'num_products', 'avg_unit_price']
].median().round(2)
profile_med['count'] = customer_df['cluster'].value_counts().sort_index()
print(profile_med)

print("\n\n")







# 6. Name segments, visualize, and save final data
# Cluster profiles determine segment names based on defining behavioral traits:
# - Regular: Baseline customers, below average on all features.
# - Bulk Buyer: Very high product variety (~4x average).
# - VIP: Highest monetary value (~2x average).
# - Premium: Highest average unit price (~2x average), buying fewer but more expensive items.
print("6. Naming segments, visualizing, and saving")


# 6.1 Assign human-readable segment names
SEGMENT_NAMES = {
    0: 'Regular',      # baseline customer, below average on all features
    1: 'Bulk Buyer',   # very high num_products (4x average)
    2: 'VIP',          # highest monetary value (~2x average)
    3: 'Premium',      # highest avg_unit_price (~2x average)
}

customer_df['segment_name'] = customer_df['cluster'].map(SEGMENT_NAMES)

print("\n   Segment sizes:")
print(customer_df['segment_name'].value_counts())


# 6.2 Final segment summary
print("\n   Segment summary (mean values per segment):")
summary = customer_df.groupby('segment_name')[
    ['monetary', 'num_products', 'avg_unit_price']
].mean().round(2)
summary['count'] = customer_df['segment_name'].value_counts()
summary['share'] = (summary['count'] / len(customer_df) * 100).round(1).astype(str) + '%'
print(summary)


# 6.3 Visualization: Pairwise scatter plots colored by segment
palette = {'Regular': '#4C72B0', 'Bulk Buyer': '#DD8452',
           'VIP': '#55A868', 'Premium': '#C44E52'}

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

feature_pairs = [
    ('monetary',     'num_products'),
    ('monetary',     'avg_unit_price'),
    ('num_products', 'avg_unit_price'),
]

for ax, (fx, fy) in zip(axes, feature_pairs):
    for seg in SEGMENT_NAMES.values():
        sub = customer_df[customer_df['segment_name'] == seg]
        ax.scatter(sub[fx], sub[fy], label=seg, alpha=0.6,
                   s=30, color=palette[seg], edgecolor='white')
    ax.set_xlabel(fx)
    ax.set_ylabel(fy)
    ax.set_title(f'{fx} vs {fy}')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

import os
os.makedirs('outputs', exist_ok=True)
plt.tight_layout()
plt.savefig('outputs/cluster_scatter_plots.png', dpi=300, bbox_inches='tight')
plt.close()


# 6.4 Save the finalized, segmented customer dataset
output_path = 'outputs/customer_segments.csv'
customer_df.to_csv(output_path, index=False)
print(f"\n   Saved: {output_path}")
print(f"   Columns: {list(customer_df.columns)}")
print(f"   Shape:   {customer_df.shape}")
print("\n\n")