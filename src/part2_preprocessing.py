import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler




# 1. Loading the dataset
print("\n1. Loading the dataset")
df = pd.read_csv('data/raw/My_retail_data.csv')
print(f"Dimensions: {df.shape[0]} rows X {df.shape[1]} columns")
print("\n\n")



# 2. Drop duplicate rows
print("\n2. Drop the duplicate rows that we detected from the EDA phase.")
df_clean = df.copy()
duplicates = df_clean.duplicated().sum()
print(f" Duplicated rows: {duplicates}")

df_clean = df_clean.drop_duplicates()

print(f" Dimensions after removal: {df_clean.shape[0]} rows X {df_clean.shape[1]} columns")
print("\n\n")




# 3. Drop useless columns from EDA findings
# Drops rule-based segments (data leakage), perfect duplicates, and near-constant columns.
# Retains only the columns necessary for building customer-level features.
print("\n3. Drop useless columns (from EDA findings)")
drop_cols = ['Customer_seg', 'Avg_UnitPrice', 'Season', 'Year',
             'Month', 'Country', 'Description']
df_clean = df_clean.drop(columns=drop_cols)
print(f" Dropped: {drop_cols}")
print(f" Remaining columns: {list(df_clean.columns)}")
print(f" Dimensions: {df_clean.shape[0]} rows X {df_clean.shape[1]} columns")
print("\n\n")








# 4. Aggregate to customer level
# Converts transaction-level data into customer-level features (one row per customer).
# Note on Recency: Excluded because all transactions occur within a 5-day window,
# giving it almost no variance and preventing meaningful segment separation.
print("\n4. Aggregating to customer level")

# Group by CustomerID and compute behavioral features
customer_df = df_clean.groupby('CustomerID').agg(
    frequency      = ('InvoiceNo',   'nunique'),   # number of unique orders
    monetary       = ('Revenue',     'sum'),       # total amount spent
    total_quantity = ('Quantity',    'sum'),       # total units purchased
    num_products   = ('StockCode',   'nunique'),   # distinct products bought
    avg_unit_price = ('UnitPrice',   'mean'),      # average price per item
).reset_index()

print(f" Customers: {customer_df.shape[0]}")
print(f" Columns: {list(customer_df.columns)}")
print("\nSample:")
print(customer_df.head())
print(f" Dimensions: {customer_df.shape[0]} rows X {customer_df.shape[1]} columns")
print("\n\n")






# 5. Remove customers with no purchases
# Some customers have 0 purchases and 0 monetary value due to fully cancelled returns.
# These rows provide no behavioral signal and would create a "phantom" cluster.
print("\n5. Removing zero-purchase customers")


# 5.1 Check affected rows
print("\n Check: counting zero-purchase customers...")

freq_zero        = (customer_df['frequency']      == 0).sum()
monetary_zero    = (customer_df['monetary']       == 0).sum()
quantity_zero    = (customer_df['total_quantity'] == 0).sum()

print(f"   Customers with frequency = 0      : {freq_zero}")
print(f"   Customers with monetary = 0       : {monetary_zero}")
print(f"   Customers with total_quantity = 0 : {quantity_zero}")

# Show a sample of the offending rows
print("\n Sample of zero-purchase customers:")
print(customer_df[customer_df['frequency'] == 0].head())


# 5.2 Remove customers with frequency = 0
# Removing on 'frequency = 0' is sufficient as it guarantees monetary = 0 and total_quantity = 0.
before = customer_df.shape[0]
customer_df = customer_df[customer_df['frequency'] > 0].reset_index(drop=True)
after = customer_df.shape[0]

print(f"\n Removed: {before - after} customers")
print(f" Remaining: {after} customers")


# 5.3 Verify no zero-purchase rows remain
print("\n Verify: checking no zero-value rows remain...")
assert (customer_df['frequency'] > 0).all(),      "Found rows with frequency = 0"
assert (customer_df['monetary'] >= 0).all(),      "Found negative monetary values"
print("   All remaining customers have at least 1 invoice.")
print(f"   New shape: {customer_df.shape[0]} rows X {customer_df.shape[1]} columns")
print("\n\n")





# 6. Descriptive statistics (customer level)
print("\n6. Descriptive statistics (customer level)")
print(customer_df.describe(include=[np.number]).round(2))

# 6.1 Inspect frequency distribution
print("\n Frequency value counts:")
print(customer_df['frequency'].value_counts().sort_index())

# 6.2 Share of customers with frequency = 1
freq1_share = (customer_df['frequency'] == 1).mean()
print(f"\n Share of customers with frequency = 1: {freq1_share:.1%}")

if freq1_share > 0.8:
    print("  Frequency is highly concentrated on a single value.")
    print("     This makes the feature near-constant and problematic")
    print("     for scaling (RobustScaler IQR = 0).")
    print("     Action: binarize to 'is_repeat' OR drop before scaling.")

print("\n\n")






# 7. Correlation check and feature selection
# Highly correlated features are double-counted in distance-based algorithms like K-Means.
# 
# Design decisions:
# - Dropped 'total_quantity': Highly correlated with 'monetary' (r = 0.86). 'monetary' 
#   is kept as it's standard in RFM and accurately captures customer engagement.
# - Dropped 'frequency': Near-constant feature (85.8% of customers have frequency = 1).
#   It cannot meaningfully separate customers and causes division-by-zero in RobustScaler.
print("\n7. Correlation check + feature selection")

# 7.1 Initial correlation check
feature_cols = ['frequency', 'monetary', 'total_quantity',
                'num_products', 'avg_unit_price']

corr = customer_df[feature_cols].corr().round(2)
print("\n Correlation matrix (before selection):")
print(corr)

print("\n Highly correlated pairs (>0.7):")
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        val = corr.iloc[i, j]
        if abs(val) > 0.7:
            print(f"   {corr.columns[i]} — {corr.columns[j]}: {val}")

# 7.2 Drop redundant and near-constant features
drop_cols = ['total_quantity', 'frequency']
customer_df = customer_df.drop(columns=drop_cols)

print(f"\n Dropped columns: {drop_cols}")
print("   - total_quantity  (redundant with monetary, r = 0.86)")
print("   - frequency       (near-constant, 85.8% = 1, IQR = 0)")

# 7.3 Final feature set
feature_cols = ['monetary', 'num_products', 'avg_unit_price']
print(f"\n Final feature set: {feature_cols}")

# 7.4 Verify no remaining high correlations
corr_final = customer_df[feature_cols].corr().round(2)
print("\n Final correlation matrix:")
print(corr_final)

print("\n Remaining highly correlated pairs (>0.7):")
remaining = False
for i in range(len(corr_final.columns)):
    for j in range(i + 1, len(corr_final.columns)):
        val = corr_final.iloc[i, j]
        if abs(val) > 0.7:
            remaining = True
            print(f"   {corr_final.columns[i]} — {corr_final.columns[j]}: {val}")
if not remaining:
    print("   No highly correlated pairs — ready for scaling.")

# 7.5 Final shape
print(f"\n Final shape: {customer_df.shape[0]} customers X {customer_df.shape[1]} columns")
print("\n Sample:")
print(customer_df.head())
print("\n\n")








# 8. Scale features with RobustScaler
# K-Means uses Euclidean distance, making scaling mandatory so features with larger
# numerical ranges (like 'monetary') do not dominate the clustering.
#
# Scaler choice: RobustScaler is used because max/median > 10 for 'monetary' and 'avg_unit_price'.
# It uses median and IQR, making it resistant to extreme VIP customer outliers,
# preserving their extreme values without compressing the middle 50% of the distribution.
print("\n8. Scaling features with RobustScaler")

from sklearn.preprocessing import RobustScaler


# 8.1 Show feature ranges before scaling
print("\n Check: feature ranges BEFORE scaling")
print(customer_df[feature_cols].agg(['min', 'max', 'median', 'std']).round(2))


# 8.2 Fit RobustScaler and transform
X = customer_df[feature_cols].copy()
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Put scaled array back into a DataFrame (keeps column names)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)


# 8.3 Verify feature stats after scaling (median ~ 0, IQR ~ 1)
print("\n Verify: feature stats AFTER scaling (median ~ 0, IQR ~ 1)")
summary = X_scaled_df.agg(['min', 'max', 'median', 'std']).round(2)
summary.loc['IQR'] = (
    X_scaled_df.quantile(0.75) - X_scaled_df.quantile(0.25)
).round(2)
print(summary)

# Confirm median ~ 0 and IQR ~ 1 for every feature
medians_ok = (X_scaled_df.median().abs() < 0.01).all()
iqr_ok     = ((X_scaled_df.quantile(0.75) - X_scaled_df.quantile(0.25) - 1).abs() < 0.01).all()
assert medians_ok, "Median of at least one feature is not ~0"
assert iqr_ok,     "IQR of at least one feature is not ~1"
print("\n   All features have median ~ 0 and IQR ~ 1.")
print(f"   Scaled shape: {X_scaled_df.shape[0]} customers X {X_scaled_df.shape[1]} features")
print("\n\n")









# 9. Saving processed data
# Saves two files: one with original values for cluster profiling, and one with
# scaled features as input for the K-Means algorithm.
print("\n9. Saving processed data")

import os
os.makedirs('data/processed', exist_ok=True)

customer_df.to_csv('data/processed/customer_level.csv', index=False)
X_scaled_df.to_csv('data/processed/customer_level_scaled.csv', index=False)

print(" Saved: data/processed/customer_level.csv")
print(" Saved: data/processed/customer_level_scaled.csv")
print(f"\n Final customer-level shape: {customer_df.shape[0]} customers X {customer_df.shape[1]} columns")
print(f" Final scaled shape:         {X_scaled_df.shape[0]} customers X {X_scaled_df.shape[1]} features")
print("\n\n")