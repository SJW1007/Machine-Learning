import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("CAR PRICE PREDICTION - REGRESSION PROBLEM")
print("Interactive Analysis with Live Visualizations")
print("=" * 80)

# =============================================================================
# STEP 1: LOAD AND PREPARE DATA
# =============================================================================
print("\n[STEP 1] Loading and Preparing Data...")

# Load File
df = pd.read_csv('used_car_listings.csv')

print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
# =============================================================================
# STEP 2: PREPROCESSING (Smart Fill)
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np

print("\n[STEP 2] Data Preprocessing...")

# Drop unnecessary columns
df_clean = df.drop(['listing_id', 'vin'], axis=1) #since id and code didn't help

# Check missing & 'Unknown' before cleaning
print("\n Checking for missing values...")
missing_info = df_clean.isnull().sum()
missing_info = missing_info[missing_info > 0].sort_values(ascending=False) #keeps only the columns that have missing values, and sorts them so the worst ones (most missing) appear first
if not missing_info.empty: #If the list isn’t empty → print what’s missing
    print(missing_info)
else:
    print("✅ No missing values found.")

print("\n Checking for 'Unknown' values...")
unknown_counts = {}
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        count = df_clean[col].astype(str).str.lower().eq('unknown').sum()
        if count > 0:
            unknown_counts[col] = count

if unknown_counts:
    for col, count in unknown_counts.items():
        print(f" - {col}: {count}")
else:
    print("✅ No 'Unknown' text values found.")

# Important columns to handle (after checking missing and unknown data)
important_cols = ['trim', 'condition', 'features']
before_rows = df_clean.shape[0] #how many before cleaning

# 📊 Missing value details before filling
missing_per_col = df_clean[important_cols].isnull().sum()
print("\n📊 Missing value details before filling:")
for col, count in missing_per_col.items():
    if count > 0:
        print(f" - {col}: {count} missing")

# 📉 Missing Row Impact Estimator (before dropping)
rows_before_drop = df_clean.shape[0]

# Count how many rows would be dropped if we drop NA from important columns
rows_with_na = df_clean[important_cols].isnull().any(axis=1).sum()
percent_missing_rows = (rows_with_na / rows_before_drop) * 100

print(f"\n🔍 {rows_with_na} rows have missing values in key columns: {important_cols}")
print(f"📉 That’s {percent_missing_rows:.2f}% of the dataset")

# Warning or note
if percent_missing_rows <= 9:
    print("✅ This is within the safe threshold (<=9%). Dropping may be acceptable.")
else:
    print("⚠️ More than 9% data loss. Consider better imputation before dropping.")


# Smart Fill: fill missing values based on most common within same make/model
print("\n Filling missing values using most common (mode) within same make/model...")
df_clean['trim'] = df_clean.groupby('model')['trim'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan)) #use the most common trim for that model.
df_clean['condition'] = df_clean.groupby('make')['condition'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan)) #based on same car make
df_clean['features'] = df_clean.groupby('model')['features'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan)) #based on same car model

# 📉 Drop any rows still missing after smart fill
df_clean.dropna(subset=important_cols, inplace=True) #if some rows still have blanks, return true
after_rows = df_clean.shape[0] #clean it
dropped_rows = before_rows - after_rows
print(f"\n🧩 Dropped {dropped_rows} rows that still had missing values after smart fill.")
# =============================================
# 📊 Outlier Detection & Visualization
# =============================================
print("\n[STEP] Outlier Detection for Real Numeric Columns")

# Select only numeric columns
numeric_cols = df_clean.select_dtypes(include=['number']).columns

# Prepare summary list
outlier_summary = []

# Loop through each real numeric column
for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df_clean[(df_clean[col] < lower_limit) | (df_clean[col] > upper_limit)]
    outlier_count = outliers.shape[0]
    total_count = df_clean.shape[0]
    outlier_percent = (outlier_count / total_count) * 100

    # Save results
    outlier_summary.append({
        'Column': col,
        'Outliers': outlier_count,
        'Percent': round(outlier_percent, 2),
        'Lower_Limit': round(lower_limit, 2),
        'Upper_Limit': round(upper_limit, 2)
    })

# Convert summary to DataFrame
outlier_df = pd.DataFrame(outlier_summary).sort_values(by='Percent', ascending=False)
print("\n📋 Outlier Summary:")
print(outlier_df)

# Visualize only the real numeric columns
# =============================================================================
# SEPARATE BOXPLOTS FOR RELEVANT OUTLIER COLUMNS (Skip Year)
# =============================================================================

# Columns you actually want to check visually
cols_to_plot = ['mileage', 'price']

print(f"\n📊 Showing boxplots for: {cols_to_plot}")

for col in cols_to_plot:
    plt.figure(figsize=(5,5))
    plt.boxplot(df_clean[col])
    plt.title(f"{col.capitalize()} Distribution (Detect Outliers)")
    plt.ylabel(col.capitalize())
    plt.show()
# =============================================================================
# 🎯 Feature Engineering: Create new helpful features
# =============================================================================
print("\n[STEP 2.9] Feature Engineering...")

# Number of features listed (from the 'features' string column)
df_clean['num_features'] = df_clean['features'].apply(
    lambda x: len(str(x).split(',')) if pd.notna(x) and x != 'Unknown' else 0
)

# Car age
df_clean['car_age'] = 2025 - df_clean['year']

print("✅ Feature engineering complete. New columns: ['num_features', 'car_age']")

# =============================================================================
# STEP 2.5: ENCODE CATEGORICAL COLUMNS
# =============================================================================
print("\n[STEP 2.5] Encoding Categorical Columns...")

# Create label encoders for each text (categorical) column
le_dict = {}
categorical_cols = ['make', 'model', 'trim', 'body_type', 'fuel_type',
                    'transmission', 'condition', 'location', 'seller_type']

for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col + '_encoded'] = le.fit_transform(df_clean[col].astype(str))
    le_dict[col] = le  # store the encoder in case you want to decode later

# Drop the original text columns
df_clean = df_clean.drop(columns=categorical_cols + ['features'])

print("✅ Encoding complete. Text columns converted to numbers.")

# =============================================================================
# STEP 3: SPLIT AND SCALE DATA
# =============================================================================
print("\n[STEP 3] Splitting and Scaling Data...")

X = df_clean.drop('price', axis=1)
y = df_clean['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

scaler = StandardScaler()#measured in the same scale for all features
X_train_scaled = scaler.fit_transform(X_train) #normalized
X_test_scaled = scaler.transform(X_test) #same rules learned from training to fix the test data too.

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ===============================
# STEP 4: RANDOM FOREST (For outlier)
# ===============================

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
# === Define IQR limits for price ===
Q1_price = df_clean['price'].quantile(0.25)
Q3_price = df_clean['price'].quantile(0.75)
IQR_price = Q3_price - Q1_price
lower_price = Q1_price - 1.5 * IQR_price
upper_price = Q3_price + 1.5 * IQR_price

# === Define IQR limits for mileage ===
Q1_mileage = df_clean['mileage'].quantile(0.25)
Q3_mileage = df_clean['mileage'].quantile(0.75)
IQR_mileage = Q3_mileage - Q1_mileage
lower_mileage = Q1_mileage - 1.5 * IQR_mileage
upper_mileage = Q3_mileage + 1.5 * IQR_mileage

# === Identify outliers ===
price_outliers = (df_clean['price'] < lower_price) | (df_clean['price'] > upper_price)
mileage_outliers = (df_clean['mileage'] < lower_mileage) | (df_clean['mileage'] > upper_mileage)

# === Combine outliers using OR ===
outliers_mask = price_outliers | mileage_outliers

# 1. Create filtered datasets
X_no_outliers = X[~outliers_mask]
y_no_outliers = y[~outliers_mask]

# 2. Copy the same test set
X_test_no_outliers = X_test.copy()
y_test_no_outliers = y_test.copy()

# 3. Fit and evaluate on full data
model_with = rf_model.fit(X, y)
r2_with = r2_score(y_test, model_with.predict(X_test))

# 4. Fit and evaluate on data without outliers
model_without = rf_model.fit(X_no_outliers, y_no_outliers)
r2_without = r2_score(y_test_no_outliers, model_without.predict(X_test_no_outliers))

# 5. Print comparison
print("\n🎯 Quick R² Comparison")
print(f"With outliers   → R²: {r2_with:.4f}")
print(f"Without outliers→ R²: {r2_without:.4f}")

if r2_without > r2_with:
    print("✅ Removing outliers helped.")
else:
    print("⚠️ Removing outliers didn’t help much.")

# ===============================
# STEP 5: NEURAL NETWORK (For outlier)
# ===============================
def build_nn(input_dim):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # output layer (regression → 1 value)
    ])

    model.compile(
        optimizer='adam',
        loss='mse',       # mean squared error for regression
        metrics=['mae']   # mean absolute error for evaluation
    )
    return model
# Define early stopping first
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 1. Scale no-outlier data
X_no_train_scaled = scaler.fit_transform(X_no_outliers)
X_test_no_scaled = scaler.transform(X_test_no_outliers)

# 2. Build and train NN on full data
nn_model_with = build_nn(X_train_scaled.shape[1])
nn_model_with.fit(X_train_scaled, y_train, validation_split=0.2,
                  epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)

r2_nn_with = r2_score(y_test, nn_model_with.predict(X_test_scaled, verbose=0).flatten())

# 3. Build and train NN on data without outliers
nn_model_without = build_nn(X_no_train_scaled.shape[1])
nn_model_without.fit(X_no_train_scaled, y_no_outliers, validation_split=0.2,
                     epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)

r2_nn_without = r2_score(y_test_no_outliers, nn_model_without.predict(X_test_no_scaled, verbose=0).flatten())

# 4. Print comparison
print("\n🧠 Quick R² Comparison (Neural Network)")
print(f"With outliers   → R²: {r2_nn_with:.4f}")
print(f"Without outliers→ R²: {r2_nn_without:.4f}")

if r2_nn_without > r2_nn_with:
    print("✅ NN improved after removing outliers.")
else:
    print("⚠️ NN did not improve much after removing outliers.")

# ===============================
# STEP 6: ENSEMBLE (For outlier)
# ===============================

# 1. Predict again using retrained models
y_pred_rf_with = model_with.predict(X_test)
y_pred_nn_with = nn_model_with.predict(X_test_scaled, verbose=0).flatten()
y_pred_ensemble_with = 0.5 * y_pred_rf_with + 0.5 * y_pred_nn_with

y_pred_rf_without = model_without.predict(X_test_no_outliers)
y_pred_nn_without = nn_model_without.predict(X_test_no_scaled, verbose=0).flatten()
y_pred_ensemble_without = 0.5 * y_pred_rf_without + 0.5 * y_pred_nn_without

# 2. Compare R²
r2_ensemble_with = r2_score(y_test, y_pred_ensemble_with)
r2_ensemble_without = r2_score(y_test_no_outliers, y_pred_ensemble_without)

# 3. Print comparison
print("\n🤖 Quick R² Comparison (Ensemble RF + NN)")
print(f"With outliers   → R²: {r2_ensemble_with:.4f}")
print(f"Without outliers→ R²: {r2_ensemble_without:.4f}")

if r2_ensemble_without > r2_ensemble_with:
    print("✅ Ensemble improved after removing outliers.")
else:
    print("⚠️ Ensemble did not improve much after removing outliers.")
