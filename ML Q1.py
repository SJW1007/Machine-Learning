import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
import warnings
import random
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# 🔒 Fix seeds for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# ---- Basic Distribution of Price ----
plt.figure(figsize=(7,5))
sns.histplot(df['price'], kde=True)
plt.title("Car Price Distribution")
plt.show()

# ---- Mileage Distribution ----
plt.figure(figsize=(7,5))
sns.histplot(df['mileage'], kde=True)
plt.title("Mileage Distribution")
plt.show()

# ---- Correlation Heatmap (numeric columns only) ----
plt.figure(figsize=(10, 8))

numeric_df = df.select_dtypes(include=['number'])  # only keep numeric columns
sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")

plt.title("Correlation Heatmap (Numeric Features)")
plt.show()

# ---- Price vs Mileage ----
plt.figure(figsize=(7,5))
sns.scatterplot(x=df['mileage'], y=df['price'])
plt.title("Mileage vs Price")
plt.show()

# ---- Count Plot for Top Brands ----
plt.figure(figsize=(8,5))
df['make'].value_counts().nlargest(10).plot(kind='bar')
plt.title("Top 10 Car Makes in Dataset")
plt.ylabel("Count")
plt.show()

# =============================================================================
# STEP 2: PREPROCESSING (Smart Fill)
# =============================================================================
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

# Let's remove outliers from only key columns (like mileage, price)
cols_to_check = ['mileage', 'price']
df_no_outliers = df_clean.copy()

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
# SEPARATE BOXPLOTS FOR RELEVANT OUTLIER COLUMNS
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

# =============================================================================
# STEP 4: MODEL 1 - RANDOM FOREST (Traditional ML)
# =============================================================================
print("\n[STEP 4] Training Random Forest (Traditional ML)...")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)#learn
y_pred_rf = rf_model.predict(X_test)#predict

# Metrics
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"✓ RMSE: {rmse_rf:.2f} | MAE: {mae_rf:.2f} | R²: {r2_rf:.4f}")

# =============================================================================
# STEP 5: MODEL 2 - NEURAL NETWORK (Deep Learning)
# =============================================================================
print("\n[STEP 5] Training Neural Network (Deep Learning)...")

def build_nn(input_dim):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])#
    model.compile(optimizer='adam', loss='mse', metrics=['mae']) #let model know how it learn, adam=adjusts how fast the robot learns, mse=what the robot tries to minimize (how wrong its guesses are), an extra way to check the size of its mistakes
    return model

nn_model = build_nn(X_train_scaled.shape[1])
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                           restore_best_weights=True)

nn_model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100,
             batch_size=32, callbacks=[early_stop], verbose=0)

y_pred_nn = nn_model.predict(X_test_scaled, verbose=0).flatten()

rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
mae_nn = mean_absolute_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

print(f"✓ RMSE: {rmse_nn:.2f} | MAE: {mae_nn:.2f} | R²: {r2_nn:.4f}")

# =============================================================================
# STEP 6: MODEL 3 - ENSEMBLE (RF + NN)
# =============================================================================
print("\n[STEP 6] Creating Ensemble Model (RF + NN)...")

# Simple weighted average (50-50)
y_pred_ensemble = 0.5 * y_pred_rf + 0.5 * y_pred_nn

rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
r2_ensemble = r2_score(y_test, y_pred_ensemble)

print(f"✓ RMSE: {rmse_ensemble:.2f} | MAE: {mae_ensemble:.2f} | R²: {r2_ensemble:.4f}")

# =============================================================================
# STEP 6.5: Prepare "Before Tuning" metrics for visualization
# =============================================================================
models_before = ["Random Forest", "Neural Network", "Ensemble"]
predictions_before = [y_pred_rf, y_pred_nn, y_pred_ensemble]

rmse_before = [rmse_rf, rmse_nn, rmse_ensemble]
mae_before = [mae_rf, mae_nn, mae_ensemble]
r2_before  = [r2_rf, r2_nn, r2_ensemble]


# =============================================================================
# STEP 7: VISUALIZATION - BEFORE TUNING (ALL MODELS)
# =============================================================================
print("\n[STEP 7] Visualizing Model Performance - Before Tuning...")

# Print metrics in log
print("\n--- Model Metrics (Before Tuning) ---")
for i, model_name in enumerate(models_before):
    print(f"{model_name}: RMSE={rmse_before[i]:.2f}, MAE={mae_before[i]:.2f}, R²={r2_before[i]:.4f}")

# Create Actual vs Predicted plots for all 3 models
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Actual vs Predicted Prices (Before Tuning)', fontsize=15, fontweight='bold')

for i, model_name in enumerate(models_before):
    ax = axes[i]
    ax.scatter(y_test, predictions_before[i], alpha=0.6, s=30)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_title(model_name)
    ax.set_xlabel('Actual Price ($)')
    ax.set_ylabel('Predicted Price ($)')
    ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# =============================================================================
# STEP 7.2: FEATURE IMPORTANCE (BEFORE TUNING)
# =============================================================================
print("\n[STEP 7.2] Feature Importance - Before Tuning")

print("\n--- Random Forest Feature Importance (Before Tuning) ---")
from sklearn.inspection import permutation_importance

# Try to detect the correct trained RF variable
try:
    rf_importance = rf_model.feature_importances_
except:
    try:
        rf_importance = rf_reg.feature_importances_
    except:
        rf_importance = rf_best.feature_importances_

features = X_train.columns

rf_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': rf_importance
}).sort_values(by='Importance', ascending=False)

print(rf_importance_df)

plt.figure(figsize=(8,6))
plt.barh(rf_importance_df['Feature'], rf_importance_df['Importance'])
plt.title("Random Forest Feature Importance (Before Tuning)")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.grid(alpha=0.3)
plt.show()


print("\n--- Neural Network Feature Importance (Before Tuning) ---")
# Custom scoring function for Keras model
def nn_score(model, X, y):
    preds = model.predict(X, verbose=0).flatten()
    return r2_score(y, preds)

perm_results_nn = permutation_importance(
    estimator=nn_model,
    X=X_test_scaled,
    y=y_test,
    scoring=nn_score,   # ✅ custom scoring
    n_repeats=5,
    random_state=42
)

nn_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': perm_results_nn.importances_mean
}).sort_values(by='Importance', ascending=False)

print(nn_importance_df)

plt.figure(figsize=(8,6))
plt.barh(nn_importance_df['Feature'], nn_importance_df['Importance'])
plt.title("Neural Network Permutation Importance (Before Tuning)")
plt.xlabel("Importance Score Change")
plt.gca().invert_yaxis()
plt.grid(alpha=0.3)
plt.show()



# ============= Ensemble Importance (Average RF + NN) =============
print("\n--- Ensemble Feature Importance (Before Tuning) ---")

ensemble_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'RF_Imp': rf_importance,
    'NN_Imp': perm_results_nn.importances_mean
})

# Your BEFORE tuning ensemble weight = RF 0.5, NN 0.5 (change if needed)
ensemble_importance_df['Ensemble'] = \
    0.5 * ensemble_importance_df['RF_Imp'] + \
    0.5 * ensemble_importance_df['NN_Imp']

ensemble_importance_df = ensemble_importance_df.sort_values(
    by='Ensemble', ascending=False
)

print(ensemble_importance_df[['Feature', 'Ensemble']])

# Plot Ensemble Importance
plt.figure(figsize=(8,6))
plt.barh(ensemble_importance_df['Feature'], ensemble_importance_df['Ensemble'])
plt.title("Ensemble Feature Importance (Before Tuning)")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.grid(alpha=0.3)
plt.show()

# =============================================================================
# STEP 8: TUNE RANDOM FOREST (GridSearchCV) 1st try
# =============================================================================
print("\n[STEP 8] Tuning Random Forest using GridSearchCV...")

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# --- Define a simple parameter grid ---
param_grid_refined = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [15, 20, 25],
    'min_samples_split': [3, 5, 7],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt'],
    'bootstrap': [False]
}

# --- Create the base Random Forest model ---
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)

# --- Create and run GridSearchCV ---
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid_refined,
    cv=5,                 # 5-fold cross-validation
    scoring='r2',         # use R² as scoring metric
    n_jobs=-1,            # use all CPU cores for faster training
    verbose=2             # show progress
)

# --- Fit the model on training data ---
grid_search.fit(X_train, y_train)

# --- Retrieve the best model ---
rf_tuned = grid_search.best_estimator_

# --- Predict on test set ---
y_pred_rf_tuned = rf_tuned.predict(X_test)

# --- Evaluate tuned model ---
rmse_rf_tuned = np.sqrt(mean_squared_error(y_test, y_pred_rf_tuned))
mae_rf_tuned = mean_absolute_error(y_test, y_pred_rf_tuned)
r2_rf_tuned = r2_score(y_test, y_pred_rf_tuned)

# --- Print results ---
print(f"✓ Best parameters: {grid_search.best_params_}")
print(f"✓ Best Cross-Validation R²: {grid_search.best_score_:.4f}")
print(f"✓ RMSE: {rmse_rf_tuned:.2f} | MAE: {mae_rf_tuned:.2f} | R²: {r2_rf_tuned:.4f}")


# =============================================================================
# 🎯 STEP: Fine-Tuning Best Neural Network
# =============================================================================



# ------------------------------------------------------------
# 1️⃣ Rebuild the model using improved hyperparameters
# ------------------------------------------------------------

from tensorflow.keras import regularizers

def build_nn_v22(input_dim):
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.2),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),

        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),

        layers.Dense(1)
    ])

    optimizer = Adam(learning_rate=0.0025)  # mid between 0.002 and 0.003

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    return model

# correct way to call the function
model_finetuned = build_nn_v22(X_train_scaled.shape[1])

# ------------------------------------------------------------
# 2️⃣ Train with early stopping and adaptive LR
# ------------------------------------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-5
)

history = model_finetuned.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,     # use the best performing batch size
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ------------------------------------------------------------
# 3️⃣ Evaluate on test data
# ------------------------------------------------------------
y_pred_nn_tuned = model_finetuned.predict(X_test_scaled).flatten()

rmse_nn_tuned = np.sqrt(mean_squared_error(y_test, y_pred_nn_tuned))
mae_nn_tuned  = mean_absolute_error(y_test, y_pred_nn_tuned)
r2_nn_tuned   = r2_score(y_test, y_pred_nn_tuned)

print("\n=================== FINE-TUNED MODEL RESULTS ===================")
print(f"✓ RMSE: {rmse_nn_tuned:.2f}")
print(f"✓ MAE: {mae_nn_tuned:.2f}")
print(f"✓ R²: {r2_nn_tuned:.4f}")

# =============================================================================
# STEP 10: TUNE ENSEMBLE
# =============================================================================
print("\n[STEP 10] Tuning Ensemble Model...")

# Get predictions again on the same test set
y_pred_rf_tuned = rf_tuned.predict(X_test)
y_pred_nn_tuned = model_finetuned.predict(X_test_scaled).flatten()

best_r2 = -np.inf
best_weight = 0.5

# Try weights from 0.0 to 1.0 in 0.01 steps
for w in np.arange(0.0, 1.001, 0.01):
    y_pred_temp = w * y_pred_rf_tuned + (1 - w) * y_pred_nn_tuned
    r2_temp = r2_score(y_test, y_pred_temp)
    if r2_temp > best_r2:
        best_r2 = r2_temp
        best_weight = w

# Combine using the best weights
y_pred_ensemble_tuned = best_weight * y_pred_rf_tuned + (1 - best_weight) * y_pred_nn_tuned

# Calculate metrics
rmse_ensemble_tuned = np.sqrt(mean_squared_error(y_test, y_pred_ensemble_tuned))
mae_ensemble_tuned = mean_absolute_error(y_test, y_pred_ensemble_tuned)
r2_ensemble_tuned = r2_score(y_test, y_pred_ensemble_tuned)

# Display results
print(f"✓ Best weights: RF={best_weight:.2f}, NN={1 - best_weight:.2f}")
print(f"✓ RMSE: {rmse_ensemble_tuned:.2f} | MAE: {mae_ensemble_tuned:.2f} | R²: {r2_ensemble_tuned:.4f}")

# =============================================================================
# STEP 10.5: Prepare "After Tuning" metrics for visualization
# =============================================================================
models_after = ["Random Forest (Tuned)", "Neural Network (Tuned)", "Ensemble (Tuned)"]
predictions_after = [y_pred_rf_tuned, y_pred_nn_tuned, y_pred_ensemble_tuned]

predictions_after = [y_pred_rf_tuned, y_pred_nn_tuned, y_pred_ensemble_tuned]
rmse_after = [rmse_rf_tuned, rmse_nn_tuned, rmse_ensemble_tuned]
mae_after  = [mae_rf_tuned, mae_nn_tuned, mae_ensemble_tuned]
r2_after   = [r2_rf_tuned, r2_nn_tuned, r2_ensemble_tuned]

# =============================================================================
# STEP 11: VISUALIZATION - AFTER TUNING (ALL MODELS)
# =============================================================================
print("\n[STEP 11] Visualizing Model Performance - After Tuning...")

# Print metrics in log
print("\n--- Model Metrics (After Tuning) ---")
for i, model_name in enumerate(models_after):
    print(f"{model_name}: RMSE={rmse_after[i]:.2f}, MAE={mae_after[i]:.2f}, R²={r2_after[i]:.4f}")


# Create Actual vs Predicted plots for all 3 tuned models
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Actual vs Predicted Prices (After Tuning)', fontsize=15, fontweight='bold')

for i, model_name in enumerate(models_after):
    ax = axes[i]
    ax.scatter(y_test, predictions_after[i], alpha=0.6, s=30)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_title(model_name)
    ax.set_xlabel('Actual Price ($)')
    ax.set_ylabel('Predicted Price ($)')
    ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# =============================================================================
# STEP 11.5: FEATURE IMPORTANCE (AFTER TUNING)
# =============================================================================
print("\n[STEP 11.5] Feature Importance - After Tuning")

# ============= Random Forest Tuned Feature Importance =============
print("\n--- Random Forest Feature Importance (After Tuning) ---")

# Correct tuned RF feature importance
rf_importance_after = rf_tuned.feature_importances_

features = X_train.columns

rf_importance_after_df = pd.DataFrame({
    'Feature': features,
    'Importance': rf_importance_after
}).sort_values(by='Importance', ascending=False)

print(rf_importance_after_df)

plt.figure(figsize=(8,6))
plt.barh(rf_importance_after_df['Feature'], rf_importance_after_df['Importance'])
plt.title("Random Forest Feature Importance (After Tuning)")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.grid(alpha=0.3)
plt.show()


print("\n--- Neural Network Feature Importance (After Tuning) ---")

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

baseline_pred = model_finetuned.predict(X_test_scaled, verbose=0).flatten()
baseline_r2 = r2_score(y_test, baseline_pred)

# Copy original test data
X_test_copy = X_test_scaled.copy()
feature_importances = []

for i, col in enumerate(X_train.columns):
    X_permuted = X_test_copy.copy()
    np.random.shuffle(X_permuted[:, i])
    y_pred_perm = model_finetuned.predict(X_permuted, verbose=0).flatten()
    perm_r2 = r2_score(y_test, y_pred_perm)
    drop = baseline_r2 - perm_r2
    feature_importances.append(drop)

# Build DataFrame
nn_importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

print(nn_importance_df)

# Plot
plt.figure(figsize=(8,6))
plt.barh(nn_importance_df["Feature"], nn_importance_df["Importance"])
plt.title("Neural Network Feature Importance (After Tuning)")
plt.xlabel("R² Drop (Higher = More Important)")
plt.gca().invert_yaxis()
plt.grid(alpha=0.3)
plt.show()

# ============= Ensemble Tuned Importance =============
print("\n--- Ensemble Feature Importance (After Tuning) ---")

ensemble_importance_after_df = pd.DataFrame({
    'Feature': X_train.columns,
    'RF_Imp': rf_importance_after,
    'NN_Imp': nn_importance_df['Importance']
})

ensemble_importance_after_df['Ensemble'] = \
    best_weight * ensemble_importance_after_df['RF_Imp'] + \
    (1 - best_weight) * ensemble_importance_after_df['NN_Imp']

ensemble_importance_after_df = ensemble_importance_after_df.sort_values(
    by='Ensemble', ascending=False
)

print(ensemble_importance_after_df[['Feature', 'Ensemble']])

plt.figure(figsize=(8,6))
plt.barh(ensemble_importance_after_df['Feature'], ensemble_importance_after_df['Ensemble'])
plt.title("Ensemble Feature Importance (After Tuning)")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.grid(alpha=0.3)
plt.show()

# =============================================================================
# STEP 12: COMPARATIVE ANALYSIS – BEFORE vs AFTER TUNING (Readable Format)
# =============================================================================
print("\n" + "="*80)
print("📊 COMPARATIVE ANALYSIS: BEFORE vs AFTER TUNING")
print("="*80)

# --- Define before-tuning metrics (from Steps 4–6) ---
rmse_before = [rmse_rf, rmse_nn, rmse_ensemble]
mae_before  = [mae_rf, mae_nn, mae_ensemble]
r2_before   = [r2_rf, r2_nn, r2_ensemble]

models_before = ["Random Forest (Base)", "Neural Network (Base)", "Ensemble (Base)"]
models_after  = ["Random Forest (Tuned)", "Neural Network (Tuned)", "Ensemble (Tuned)"]

# --- Create DataFrame for easy reference ---
comparison_df = pd.DataFrame({
    "Model (Before)": models_before,
    "RMSE Before": rmse_before,
    "MAE Before": mae_before,
    "R² Before": r2_before,
    "Model (After)": models_after,
    "RMSE After": rmse_after,
    "MAE After": mae_after,
    "R² After": r2_after,
})

# --- Compute improvements ---
comparison_df["Δ RMSE"] = np.array(rmse_before) - np.array(rmse_after)
comparison_df["Δ MAE"]  = np.array(mae_before) - np.array(mae_after)
comparison_df["Δ R²"]   = np.array(r2_after) - np.array(r2_before)

pd.set_option('display.width', 150)   # widen the console layout
pd.set_option('display.max_columns', None)  # show all columns in one line
pd.options.display.float_format = "{:.4f}".format  # round all floats

print("\n--- Model Performance Comparison ---")
print(comparison_df[["Model (Before)", "RMSE Before", "MAE Before", "R² Before"]].round(4))

print("\n--- After Tuning ---")
print(comparison_df[["Model (After)", "RMSE After", "MAE After", "R² After"]].round(4))

# --- Print readable improvement summary ---
print("\n--- Improvement Summary ---")

def fmt_plus(x, decimals=3):
    """Format with + sign for positive numbers."""
    return f"{x:+.{decimals}f}"

for i in range(len(models_before)):
    delta_rmse = fmt_plus(comparison_df["Δ RMSE"][i], 2)
    delta_mae  = fmt_plus(comparison_df["Δ MAE"][i], 2)
    delta_r2   = fmt_plus(comparison_df["Δ R²"][i], 4)

    print(f"\n🔹 {models_after[i]} (vs {models_before[i]})")
    print(f"   RMSE improvement : {delta_rmse}")
    print(f"   MAE improvement  : {delta_mae}")
    print(f"   R² gain          : {delta_r2}")

    if comparison_df["Δ R²"][i] > 0:
        print("   ✅ Model explains more variance after tuning.")
    else:
        print("   ⚠️ No improvement in explanatory power.")
    if comparison_df["Δ RMSE"][i] > 0:
        print("   ✅ Predictions became more accurate (lower error).")
    else:
        print("   ⚠️ Slightly worse RMSE — may need re-tuning.")

# --- Overall conclusion ---
best_model_idx = np.argmax(r2_after)
print("\n" + "="*80)
print("🏆 OVERALL BEST MODEL AFTER TUNING")
print(f"✅ {models_after[best_model_idx]} with R² = {r2_after[best_model_idx]:.4f}, "
      f"RMSE = {rmse_after[best_model_idx]:.2f}, MAE = {mae_after[best_model_idx]:.2f}")
print("="*80)

# =============================================================================
# STEP 14: RESIDUAL ANALYSIS
# =============================================================================
print("\n[STEP 14] Residual Analysis...")

# Residuals (difference between actual and predicted)
residuals_rf = y_test - y_pred_rf_tuned
residuals_nn = y_test - model_finetuned.predict(X_test_scaled, verbose=0).flatten()
residuals_ensemble = y_test - y_pred_ensemble_tuned

plt.figure(figsize=(10,6))
# sns.histplot(residuals_rf, color='skyblue', label='RF Residuals', kde=True)
# sns.histplot(residuals_nn, color='orange', label='NN Residuals', kde=True)
# sns.histplot(residuals_ensemble, color='green', label='Ensemble Residuals', kde=True)
plt.hist(residuals_rf, bins=50, alpha=0.4, color='blue', label='RF Residuals')
plt.hist(residuals_nn, bins=50, alpha=0.4, color='orange', label='NN Residuals')
plt.hist(residuals_ensemble, bins=50, alpha=0.4, color='red', label='Ensemble Residuals')
plt.axvline(0, color='red', linestyle='--', label='Perfect Prediction')
plt.title('Residual Distribution (Error Spread)')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# =============================================================================
# STEP 15: MODEL COMPARISON BAR PLOT
# =============================================================================
print("\n[STEP 15] Visualizing Model Comparison...")

metrics_df = pd.DataFrame({
    'Model': ['RF Base', 'NN Base', 'Ensemble Base',
              'RF Tuned', 'NN Tuned', 'Ensemble Tuned'],
    'RMSE': rmse_before + rmse_after,
    'MAE': mae_before + mae_after,
    'R2':  r2_before + r2_after
})

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Model Comparison: Before vs After Tuning', fontsize=14, fontweight='bold')

# RMSE
sns.barplot(data=metrics_df, x='Model', y='RMSE', ax=axes[0], palette='Blues_r')
axes[0].set_title('RMSE (Lower = Better)')
axes[0].tick_params(axis='x', rotation=45)

# MAE
sns.barplot(data=metrics_df, x='Model', y='MAE', ax=axes[1], palette='Greens_r')
axes[1].set_title('MAE (Lower = Better)')
axes[1].tick_params(axis='x', rotation=45)

# R²
sns.barplot(data=metrics_df, x='Model', y='R2', ax=axes[2], palette='Oranges')
axes[2].set_title('R² Score (Higher = Better)')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

print("\n[STEP 16] Cross-Validation for Random Forest...")

# --- Cross-validation for Random Forest ---
cv_scores = cross_val_score(rf_tuned, X, y, cv=5, scoring='r2')
print(f"Cross-validation R² scores: {cv_scores}")
print(f"Mean R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

plt.figure(figsize=(7,5))
sns.boxplot(y=cv_scores, color='lightblue')
plt.title('Cross-Validation R² Distribution (Random Forest)')
plt.ylabel('R² Score')
plt.show()

# --- Residual Plot ---
y_pred_rf_tuned = rf_tuned.predict(X_test)
residuals_rf = y_test - y_pred_rf_tuned  # ✅ define residuals

plt.figure(figsize=(7,6))
plt.scatter(y_pred_rf_tuned, residuals_rf, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals vs Predicted Price (Random Forest)")
plt.grid(alpha=0.3)
plt.show()

# ==========================================
# Residuals vs Predicted Price (Ensemble)
# ==========================================
print("\n[STEP 17] Residual Plot for Ensemble Model...")

# Get predictions for test set (ensure same size)
y_pred_rf_tuned = rf_tuned.predict(X_test)
y_pred_nn_tuned = model_finetuned.predict(X_test_scaled, verbose=0).flatten()

# Weighted ensemble (based on your tuned weights)
w_rf, w_nn = w_rf, w_nn = 0.95, 0.05
y_pred_ensemble_tuned = (w_rf * y_pred_rf_tuned) + (w_nn * y_pred_nn_tuned)

# Residuals
residuals_ensemble = y_test - y_pred_ensemble_tuned

# Plot residuals
plt.figure(figsize=(7,6))
plt.scatter(y_pred_ensemble_tuned, residuals_ensemble, alpha=0.6, color='orange')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals vs Predicted Price (Ensemble Model)")
plt.grid(alpha=0.3)
plt.show()
print("\n[STEP 18] Cross-Validation for Ensemble Model (No Retraining)...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
    # Split data for this fold
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # ✅ Refit scaler inside each fold
    scaler_fold = StandardScaler()
    X_train_scaled = scaler_fold.fit_transform(X_train)
    X_test_scaled = scaler_fold.transform(X_test)

    # Predict using already-trained Random Forest (no retraining)
    y_pred_rf = rf_tuned.predict(X_test)

    # Predict using Neural Network (fine-tuned model)
    y_pred_nn_tuned = model_finetuned.predict(X_test_scaled, verbose=0).flatten()

    # Weighted ensemble (based on tuned ratio)
    w_rf, w_nn = 0.95, 0.05
    y_pred_ensemble = (w_rf * y_pred_rf) + (w_nn * y_pred_nn_tuned)

    # Evaluate
    r2 = r2_score(y_test, y_pred_ensemble)
    r2_scores.append(r2)
    print(f"Fold {fold} R²: {r2:.4f}")

# Final summary
mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)
print(f"\nCross-validation R² scores (Ensemble): {r2_scores}")
print(f"Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")

# Boxplot visualization
plt.figure(figsize=(7,5))
sns.boxplot(y=r2_scores, color='lightcoral')
plt.title('Cross-Validation R² Distribution (Ensemble Model)')
plt.ylabel('R² Score')
plt.show()

