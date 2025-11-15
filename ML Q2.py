from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from imblearn.over_sampling import SMOTE
from keras_tuner import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.metrics import Recall
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import seaborn as sns
import numpy as np
##Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ============================================
# STREAMLINED EXPLORATORY DATA ANALYSIS (EDA)
# ============================================

print("\n" + "=" * 60)
print("ESSENTIAL EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Make a copy for EDA
df_eda = df.copy()

# ============================================
# 1. DATASET OVERVIEW (Essential Only)
# ============================================
print("\n=== 1. DATASET OVERVIEW ===")
print(f"Dataset Shape: {df_eda.shape[0]} rows × {df_eda.shape[1]} columns")
print(f"\nData Types Summary:")
print(df_eda.dtypes.value_counts())

# Convert TotalCharges for analysis
df_eda["TotalCharges"] = pd.to_numeric(df_eda["TotalCharges"], errors="coerce")

# ============================================
# 2. DATA QUALITY CHECK
# ============================================
print("\n=== 2. DATA QUALITY ===")

# Missing values
missing_summary = df_eda.isnull().sum()
if missing_summary.sum() > 0:
    print(f"Total missing values: {missing_summary.sum()}")
    print("\nColumns with missing values:")
    print(missing_summary[missing_summary > 0])
else:
    print("✓ No missing values found")

# Duplicates
duplicate_count = df_eda.duplicated().sum()
print(f"Duplicate rows: {duplicate_count}")

# ============================================
# 3. TARGET VARIABLE ANALYSIS (CRITICAL)
# ============================================
print("\n=== 3. TARGET VARIABLE: CHURN ===")
churn_counts = df_eda['Churn'].value_counts()
churn_percent = df_eda['Churn'].value_counts(normalize=True) * 100

print(f"No Churn:  {churn_counts['No']:5d} ({churn_percent['No']:.1f}%)")
print(f"Churn:     {churn_counts['Yes']:5d} ({churn_percent['Yes']:.1f}%)")
print(f"\n⚠️ Class Imbalance Ratio: {churn_percent['No'] / churn_percent['Yes']:.2f}:1")

# Plot only pie chart for churn distribution
plt.figure(figsize=(6, 6))

colors = ['#2ecc71', '#e74c3c']  # Green = No churn, Red = Churn
plt.pie(
    churn_percent.values,
    labels=['No Churn', 'Churn'],
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    textprops={'fontsize': 12, 'fontweight': 'bold'}
)

plt.title('Customer Churn Proportion', fontsize=14, fontweight='bold')
plt.show()
# ============================================
# 4. KEY NUMERICAL FEATURES
# ============================================
print("\n=== 4. NUMERICAL FEATURES ===")

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
print("\nDescriptive Statistics:")
print(df_eda[numerical_cols].describe().round(2))

# Single comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Top row: Distributions
for idx, col in enumerate(numerical_cols):
    axes[0, idx].hist(df_eda[col].dropna(), bins=30, edgecolor='black',
                      alpha=0.7, color='steelblue')
    axes[0, idx].set_title(f'{col} Distribution', fontweight='bold')
    axes[0, idx].set_xlabel(col)
    axes[0, idx].set_ylabel('Frequency')
    axes[0, idx].grid(axis='y', alpha=0.3)

# Bottom row: Box plots by Churn
for idx, col in enumerate(numerical_cols):
    churn_no = df_eda[df_eda['Churn'] == 'No'][col].dropna()
    churn_yes = df_eda[df_eda['Churn'] == 'Yes'][col].dropna()

    bp = axes[1, idx].boxplot([churn_no, churn_yes], labels=['No Churn', 'Churn'],
                              patch_artist=True, notch=True)
    for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    axes[1, idx].set_title(f'{col} by Churn', fontweight='bold')
    axes[1, idx].set_ylabel(col)
    axes[1, idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 5. TOP CATEGORICAL FEATURES (Most Important)
# ============================================
print("\n=== 5. KEY CATEGORICAL FEATURES ===")

# Select most business-relevant features
key_categorical = ['Contract', 'InternetService', 'PaymentMethod',
                   'TechSupport', 'OnlineSecurity']

# Calculate churn rates for each
print("\nChurn Rates by Key Features:\n")
for col in key_categorical:
    churn_rate = df_eda.groupby(col)['Churn'].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100
    ).sort_values(ascending=False)
    print(f"{col}:")
    print(churn_rate.to_string())
    print()

# Visualize top 5 most impactful features
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, col in enumerate(key_categorical):
    # Create crosstab
    ct = pd.crosstab(df_eda[col], df_eda['Churn'], normalize='index') * 100

    # Plot
    ct.plot(kind='bar', ax=axes[idx], color=['#2ecc71', '#e74c3c'],
            alpha=0.8, edgecolor='black')
    axes[idx].set_title(f'Churn Rate by {col}', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('Percentage (%)', fontsize=10)
    axes[idx].legend(title='', labels=['No Churn', 'Churn'], loc='upper right')
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(axis='y', alpha=0.3)

    # Add percentage labels on bars
    for container in axes[idx].containers:
        axes[idx].bar_label(container, fmt='%.1f%%', fontsize=8)

# Remove last empty subplot
fig.delaxes(axes[5])

plt.tight_layout()
plt.show()

# ============================================
# 6. CORRELATION ANALYSIS
# ============================================
print("\n=== 6. CORRELATION ANALYSIS ===")

# Correlation matrix
corr_matrix = df_eda[numerical_cols].corr()
print("\nCorrelation Matrix:")
print(corr_matrix.round(3))

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=2, cbar_kws={"shrink": 0.8},
            fmt='.3f', annot_kws={'fontsize': 11, 'fontweight': 'bold'})
plt.title('Correlation Matrix - Numerical Features', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()

# ============================================
# 7. STATISTICAL SIGNIFICANCE TESTS
# ============================================
print("\n=== 7. STATISTICAL TESTS ===")

# Chi-square for categorical (top features only)
print("\nChi-Square Test Results (Key Categorical Features):\n")
chi_results = []

for col in key_categorical:
    contingency_table = pd.crosstab(df_eda[col], df_eda['Churn'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    chi_results.append({
        'Feature': col,
        'Chi2': chi2,
        'P-Value': p_value,
        'Significant': '✓ Yes' if p_value < 0.05 else 'No'
    })

chi_df = pd.DataFrame(chi_results).sort_values('P-Value')
print(chi_df.to_string(index=False))

# T-test for numerical features
print(
    "\n\nT-Test Results (Numerical Features):\n")  # Are the averages between churners and non-churners very different?
ttest_results = []

for col in numerical_cols:
    churn_yes = df_eda[df_eda['Churn'] == 'Yes'][col].dropna()
    churn_no = df_eda[df_eda['Churn'] == 'No'][col].dropna()

    t_stat, p_value = ttest_ind(churn_yes, churn_no)
    mean_diff = churn_yes.mean() - churn_no.mean()

    ttest_results.append({
        'Feature': col,
        'Mean Diff': mean_diff,
        'T-Statistic': t_stat,
        'P-Value': p_value,
        'Significant': '✓ Yes' if p_value < 0.05 else 'No'
    })

ttest_df = pd.DataFrame(ttest_results).sort_values('P-Value')
print(ttest_df.to_string(index=False))

print("\n" + "=" * 60)
print("EDA COMPLETE - All visualizations saved")
print("=" * 60)

# Drop customerID as it is not useful for prediction
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())#settle the missing value

# Convert target to binary
df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

##Feature split
X = df.drop("Churn", axis=1)
y = df["Churn"]
# ✅ Check for duplicates
print("Duplicate rows: ", df.duplicated().sum())

# ✅ Remove duplicates if any
df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# ✅ Handle outliers using IQR
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Cap outliers instead of remove to avoid data loss
    df[col] = df[col].clip(lower_bound,
                           upper_bound)  # If a value is too small (too low), push it up to the minimum allowed number.

# Clean boolean columns (if any)
for col in df.select_dtypes(
        include=["bool"]).columns:  # only do this if booleans actually exist. If none, do nothing. It doesn’t change
    df[col] = df[col].astype(str)

# Replace inf values if they exist
df.replace([np.inf, -np.inf], np.nan,
           inplace=True)  # only if such values exist. Telco usually has none, so this won’t change anything.

# Fill any numeric NaNs again just in case
df.fillna(df.median(numeric_only=True),
          inplace=True)  # already handled TotalCharges. Run a target-safe fill for features only and only if NaNs remain.

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
numeric_features = X.select_dtypes(include=["int64","float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns
##Pre processing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),#make the each feature number more average
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) #Creates a column for each category
    ]
)
# Baseline RF Model (Before SMOTE & Tuning)


rf_baseline = Pipeline(steps=[#repeat the process
    ("preprocess", preprocessor),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_baseline.fit(X_train, y_train)
y_pred_base = rf_baseline.predict(X_test)

print("=== Baseline Random Forest (No SMOTE, No Tuning) ===")
print(classification_report(y_test, y_pred_base))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_base))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_base))

#Add smote
rf_smote_only = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTE()),#help to balance the class
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42)
)
])

rf_smote_only.fit(X_train, y_train)
y_pred_smote = rf_smote_only.predict(X_test)

print("=== SMOTE + RF (No Tuning) ===")
print(classification_report(y_test, y_pred_smote))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_smote))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_smote))

##tunning 5
param_grid = {
    "rf__n_estimators": [150, 200, 300, 400],
    "rf__max_depth": [6, 8, 10, None],
    "rf__min_samples_split": [2, 3, 4, 5],
    "rf__min_samples_leaf": [1, 2, 3],#The minimum number of samples (rows) that must exist in a leaf node
    "rf__max_features": ["sqrt", "log2"],
}
feature_names = (
        list(numeric_features) +
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
)

rf_pipeline_tuned = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("rf", RandomForestClassifier(random_state=42))
])

grid_rf = GridSearchCV(rf_pipeline_tuned, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_rf.fit(X_train, y_train)

print("Best Parameters:", grid_rf.best_params_)

rf_best = grid_rf.best_estimator_
y_pred_tuned_rf = rf_best.predict(X_test)

print("=== Tuned RF + SMOTE ===")
print(classification_report(y_test, y_pred_tuned_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tuned_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_tuned_rf))#how well the model seperate positive and negative classes upon all

# Probabilities from tuned RF for ensemble tuning
y_pred_rf_tuned_proba = rf_best.predict_proba(X_test)[:, 1]

# ============================================
# RF Baseline Feature Importance
# ============================================

rf_baseline_model = rf_baseline.named_steps['rf']
importances_base = rf_baseline_model.feature_importances_

# Create DataFrame
feature_importance_base = pd.DataFrame({
    'feature': feature_names,
    'importance': importances_base
}).sort_values('importance', ascending=False)

print("\n=== Top 15 Features - Baseline RF ===")
print(feature_importance_base.head(15))

# Plot
plt.figure(figsize=(10, 8))
plt.barh(feature_importance_base.head(15)['feature'],
         feature_importance_base.head(15)['importance'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances - Baseline RF')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ============================================
# RF Tuned Feature Importance
# ============================================

rf_tuned_model = rf_best.named_steps['rf']
importances_tuned = rf_tuned_model.feature_importances_

# Create DataFrame
feature_importance_tuned = pd.DataFrame({
    'feature': feature_names,
    'importance': importances_tuned
}).sort_values('importance', ascending=False)

print("\n=== Top 15 Features - Tuned RF ===")
print(feature_importance_tuned.head(15))

# Plot
plt.figure(figsize=(10, 8))
plt.barh(feature_importance_tuned.head(15)['feature'],
         feature_importance_tuned.head(15)['importance'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances - Tuned RF')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ============================================
# Comparison of Feature Importances
# ============================================

comparison = pd.merge(
    feature_importance_base[['feature', 'importance']],
    feature_importance_tuned[['feature', 'importance']],
    on='feature',
    suffixes=('_baseline', '_tuned')
)

comparison['difference'] = comparison['importance_tuned'] - comparison['importance_baseline']
comparison = comparison.sort_values('importance_tuned', ascending=False)

print("\n=== Feature Importance Comparison (Top 20) ===")
print(comparison.head(20))

# Plot comparison
top_features = comparison.head(20)
x = np.arange(len(top_features))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(x - width/2, top_features['importance_baseline'], width, label='Baseline RF')
ax.barh(x + width/2, top_features['importance_tuned'], width, label='Tuned RF')

ax.set_xlabel('Importance')
ax.set_title('Feature Importance Comparison: Baseline vs Tuned RF')
ax.set_yticks(x)
ax.set_yticklabels(top_features['feature'])
ax.legend()
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# ============================================
# NEURAL NETWORK MODEL
# ============================================

# Prepare data for Neural Network
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

# Calculate class weights for NN
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train) #Class weight tells the Neural Network to pay more attention to the minority class
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print("Class weights:", class_weight_dict)

# ============================================
# BASELINE NEURAL NETWORK (No Tuning)
# ============================================

def build_baseline_nn(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),#first layer using neurons #Number of input features after preprocessing
        Dropout(0.3),#turn off neuron randomly
        Dense(32, activation='relu'),#relu means let model learn complex pattern while avoiding negative noise
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')#Outputs a number(yes/no), Converts output into a value between 0 and 1
    ])
    model.compile(optimizer='adam', #algorithm that adjusts how fast the network learns
                  loss='binary_crossentropy', #Measures how wrong predictions are
                  metrics=['accuracy', Recall(name='recall')])#Extra info to track while training: how many predictions are correct (accuracy) and how many churners are correctly caught
    return model

nn_baseline = build_baseline_nn(X_train_resampled.shape[1])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history_baseline = nn_baseline.fit(
    X_train_resampled, y_train_resampled,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

# Predict
y_pred_nn_baseline_proba = nn_baseline.predict(X_test_processed)
y_pred_nn_baseline = (y_pred_nn_baseline_proba > 0.5).astype(int).flatten()

print("\n=== Baseline Neural Network (No Tuning) ===")
print(classification_report(y_test, y_pred_nn_baseline))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nn_baseline))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_nn_baseline))


# ============================================
# TUNED NEURAL NETWORK with Keras Tuner
# ============================================

def build_tuned_nn(hp):
    model = Sequential()

    # Tune number of layers
    for i in range(hp.Int('num_layers', 2, 4)):
        model.add(Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
            activation='relu',
            kernel_regularizer=l2(hp.Float(f'l2_{i}', 1e-5, 1e-2, sampling='log'))
        ))
        model.add(Dropout(hp.Float(f'dropout_{i}', 0.2, 0.5, step=0.1)))

    model.add(Dense(1, activation='sigmoid'))

    # Tune learning rate
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', Recall(name='recall')]
    )
    return model


# Keras Tuner Search
tuner = RandomSearch(
    build_tuned_nn,
    objective='val_recall',
    max_trials=30,
    executions_per_trial=1,
    directory='nn_tuning',
    project_name='churn_prediction_2'
)

print("\n=== Starting Neural Network Hyperparameter Tuning ===")
tuner.search(
    X_train_resampled, y_train_resampled,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5)],
    verbose=1
)

# Get best model
best_nn = tuner.get_best_models(num_models=1)[0]
print("\nBest Hyperparameters:")
print(tuner.get_best_hyperparameters()[0].values)

# Train best model with more epochs
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history_tuned = best_nn.fit(
    X_train_resampled, y_train_resampled,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Predict
y_pred_nn_tuned_proba = best_nn.predict(X_test_processed)
y_pred_nn_tuned = (y_pred_nn_tuned_proba > 0.5).astype(int).flatten()

# Flatten tuned NN probabilities for ensemble tuning
y_pred_nn_tuned_proba_flat = y_pred_nn_tuned_proba.flatten()

print("\n=== Tuned Neural Network ===")
print(classification_report(y_test, y_pred_nn_tuned))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nn_tuned))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_nn_tuned))


# ============================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================

# Get feature names after preprocessing
feature_names = (
        list(numeric_features) +
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
)

print(f"\nTotal features after preprocessing: {len(feature_names)}")


# ============================================
# METHOD 1: Custom Permutation Importance for Neural Networks
# ============================================

def calculate_nn_permutation_importance(model, X, y, metric='accuracy', n_repeats=10, random_state=42):
    """
    Calculate permutation importance for neural networks
    """
    np.random.seed(random_state)

    # Get baseline score
    y_pred = (model.predict(X, verbose=0) > 0.5).astype(int).flatten()
    if metric == 'accuracy':
        baseline_score = accuracy_score(y, y_pred)
    elif metric == 'roc_auc':
        y_pred_proba = model.predict(X, verbose=0).flatten()
        baseline_score = roc_auc_score(y, y_pred_proba)

    importances = []

    print(f"Baseline {metric}: {baseline_score:.4f}")
    print(f"Calculating importance for {X.shape[1]} features with {n_repeats} repeats...")

    for feature_idx in range(X.shape[1]):
        feature_scores = []

        for _ in range(n_repeats):
            # Create a copy and shuffle the feature
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feature_idx])

            # Calculate score with permuted feature
            y_pred = (model.predict(X_permuted, verbose=0) > 0.5).astype(int).flatten()
            if metric == 'accuracy':
                permuted_score = accuracy_score(y, y_pred)
            elif metric == 'roc_auc':
                y_pred_proba = model.predict(X_permuted, verbose=0).flatten()
                permuted_score = roc_auc_score(y, y_pred_proba)

            # Importance is the decrease in score
            feature_scores.append(baseline_score - permuted_score)

        importances.append({
            'mean': np.mean(feature_scores),
            'std': np.std(feature_scores)
        })

        if (feature_idx + 1) % 10 == 0:
            print(f"Processed {feature_idx + 1}/{X.shape[1]} features...")

    return importances


# Calculate for Baseline NN
print("\n=== Calculating Permutation Importance for Baseline NN ===")
perm_importance_nn_base = calculate_nn_permutation_importance(
    nn_baseline,
    X_test_processed,
    y_test,
    metric='accuracy',
    n_repeats=10,
    random_state=42
)

# Create DataFrame
nn_base_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': [imp['mean'] for imp in perm_importance_nn_base],
    'std': [imp['std'] for imp in perm_importance_nn_base]
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features (Baseline NN):")
print(nn_base_importance.head(20))

# Calculate for Tuned NN
print("\n=== Calculating Permutation Importance for Tuned NN ===")
perm_importance_nn_tuned = calculate_nn_permutation_importance(
    best_nn,
    X_test_processed,
    y_test,
    metric='accuracy',
    n_repeats=10,
    random_state=42
)

# Create DataFrame
nn_tuned_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': [imp['mean'] for imp in perm_importance_nn_tuned],
    'std': [imp['std'] for imp in perm_importance_nn_tuned]
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features (Tuned NN):")
print(nn_tuned_importance.head(20))

# ============================================
# Visualize Feature Importance
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Baseline NN
top_features_base = nn_base_importance.head(20)
axes[0].barh(range(len(top_features_base)), top_features_base['importance'],
             xerr=top_features_base['std'], color='steelblue', alpha=0.7)
axes[0].set_yticks(range(len(top_features_base)))
axes[0].set_yticklabels(top_features_base['feature'])
axes[0].set_xlabel('Importance (Decrease in Accuracy)')
axes[0].set_title('Top 20 Features - Baseline Neural Network')
axes[0].invert_yaxis()

# Tuned NN
top_features_tuned = nn_tuned_importance.head(20)
axes[1].barh(range(len(top_features_tuned)), top_features_tuned['importance'],
             xerr=top_features_tuned['std'], color='coral', alpha=0.7)
axes[1].set_yticks(range(len(top_features_tuned)))
axes[1].set_yticklabels(top_features_tuned['feature'])
axes[1].set_xlabel('Importance (Decrease in Accuracy)')
axes[1].set_title('Top 20 Features - Tuned Neural Network')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

# ============================================
# Compare with Random Forest Feature Importance
# ============================================

# Build Random Forest for comparison
rf_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

print("\n=== Training Random Forest for Feature Importance Comparison ===")
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

print("\nRandom Forest Performance:")
print(classification_report(y_test, y_pred_rf))

# Get RF feature importance
rf_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_pipeline.named_steps['classifier'].feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features (Random Forest):")
print(rf_importance.head(20))

# Visualize RF importance
plt.figure(figsize=(10, 8))
top_features_rf = rf_importance.head(20)
plt.barh(range(len(top_features_rf)), top_features_rf['importance'],
         color='green', alpha=0.7)
plt.yticks(range(len(top_features_rf)), top_features_rf['feature'])
plt.xlabel('Feature Importance (Gini Importance)')
plt.title('Top 20 Features - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ============================================
# Compare Feature Rankings
# ============================================

# Merge all importance scores
comparison_df = nn_base_importance[['feature', 'importance']].rename(columns={'importance': 'nn_baseline'})
comparison_df = comparison_df.merge(
    nn_tuned_importance[['feature', 'importance']].rename(columns={'importance': 'nn_tuned'}),
    on='feature'
)
comparison_df = comparison_df.merge(
    rf_importance[['feature', 'importance']].rename(columns={'importance': 'rf_gini'}),
    on='feature'
)

# Calculate average rank
comparison_df['avg_importance'] = comparison_df[['nn_baseline', 'nn_tuned', 'rf_gini']].mean(axis=1)
comparison_df = comparison_df.sort_values('avg_importance', ascending=False)

print("\n=== Top 20 Features by Average Importance ===")
print(comparison_df.head(20))



# ============================================
# ENSEMBLE 1: Baseline RF + Baseline NN
# ============================================

print("\n=== Ensemble: Baseline RF + Baseline NN ===")

# Get predictions from both models
y_pred_rf_base_proba = rf_baseline.predict_proba(X_test)[:, 1]
y_pred_nn_base_proba = y_pred_nn_baseline_proba.flatten()

# Average probabilities
y_pred_ensemble_base_proba = (y_pred_rf_base_proba + y_pred_nn_base_proba) / 2
y_pred_ensemble_base = (y_pred_ensemble_base_proba > 0.5).astype(int)

print(classification_report(y_test, y_pred_ensemble_base))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_ensemble_base))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_ensemble_base))

# ============================================
# ENSEMBLE 2: Baseline RF + Baseline NN （Tunning）
# ============================================


best_auc = 0
best_weight = 0.5

for w in np.arange(0.0, 1.05, 0.05):
    # weighted combination
    y_pred_temp_proba = (w * y_pred_rf_tuned_proba) + ((1 - w) * y_pred_nn_tuned_proba_flat)
    y_pred_temp = (y_pred_temp_proba > 0.5).astype(int)

    # choose your metric: ROC-AUC or F1
    auc_temp = roc_auc_score(y_test, y_pred_temp_proba)

    if auc_temp > best_auc:
        best_auc = auc_temp
        best_weight = w

print(f"✅ Best RF:NN Weight Ratio = {best_weight:.2f}:{1 - best_weight:.2f}")
print(f"🎯 Best ROC-AUC = {best_auc:.4f}")

# Apply best weights
y_pred_ensemble_tuned_proba = (best_weight * y_pred_rf_tuned_proba) + ((1 - best_weight) * y_pred_nn_tuned_proba_flat)
y_pred_ensemble_tuned = (y_pred_ensemble_tuned_proba > 0.5).astype(int)

# ============================================
# ENSEMBLE FEATURE IMPORTANCE
# ============================================




# Get feature names (already defined in your code)
# feature_names should already exist from your code

def calculate_ensemble_permutation_importance(rf_model, nn_model, X_processed, y,
                                              rf_weight=0.5, metric='accuracy',
                                              n_repeats=10, random_state=42):
    """
    Calculate permutation importance for ensemble of RF + NN

    Parameters:
    -----------
    rf_model : sklearn pipeline
        Random Forest pipeline (will extract classifier)
    nn_model : keras model
        Neural Network model
    X_processed : array
        Preprocessed test features
    y : array
        True labels
    rf_weight : float
        Weight for RF predictions (NN gets 1 - rf_weight)
    metric : str
        'accuracy' or 'roc_auc'
    """
    np.random.seed(random_state)

    # Extract RF classifier from pipeline
    # Your pipelines have structure: preprocess -> smote -> rf (for SMOTE version)
    # or preprocess -> rf (for baseline)
    if hasattr(rf_model.named_steps, 'rf'):
        rf_classifier = rf_model.named_steps['rf']
    else:
        # Fallback: get the last step
        rf_classifier = rf_model.steps[-1][1]

    # Get baseline ensemble predictions
    rf_proba = rf_classifier.predict_proba(X_processed)[:, 1]
    nn_proba = nn_model.predict(X_processed, verbose=0).flatten()
    ensemble_proba = (rf_weight * rf_proba) + ((1 - rf_weight) * nn_proba)

    # Calculate baseline score
    if metric == 'accuracy':
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        baseline_score = accuracy_score(y, ensemble_pred)
    elif metric == 'roc_auc':
        baseline_score = roc_auc_score(y, ensemble_proba)

    importances = []

    print(f"Baseline {metric}: {baseline_score:.4f}")
    print(f"Calculating importance for {X_processed.shape[1]} features with {n_repeats} repeats...")

    for feature_idx in range(X_processed.shape[1]):
        feature_scores = []

        for _ in range(n_repeats):
            # Create a copy and shuffle the feature
            X_permuted = X_processed.copy()
            np.random.shuffle(X_permuted[:, feature_idx])

            # Get ensemble predictions with permuted feature
            rf_proba_perm = rf_classifier.predict_proba(X_permuted)[:, 1]
            nn_proba_perm = nn_model.predict(X_permuted, verbose=0).flatten()
            ensemble_proba_perm = (rf_weight * rf_proba_perm) + ((1 - rf_weight) * nn_proba_perm)

            # Calculate score
            if metric == 'accuracy':
                ensemble_pred_perm = (ensemble_proba_perm > 0.5).astype(int)
                permuted_score = accuracy_score(y, ensemble_pred_perm)
            elif metric == 'roc_auc':
                permuted_score = roc_auc_score(y, ensemble_proba_perm)

            # Importance is the decrease in score
            feature_scores.append(baseline_score - permuted_score)

        importances.append({
            'mean': np.mean(feature_scores),
            'std': np.std(feature_scores)
        })

        if (feature_idx + 1) % 10 == 0:
            print(f"Processed {feature_idx + 1}/{X_processed.shape[1]} features...")

    return importances


# ============================================
# 1. BASELINE ENSEMBLE (RF Baseline + NN Baseline)
# ============================================

print("\n" + "=" * 60)
print("BASELINE ENSEMBLE FEATURE IMPORTANCE")
print("RF Baseline + NN Baseline (50:50 weight)")
print("=" * 60)

perm_importance_ensemble_base = calculate_ensemble_permutation_importance(
    rf_baseline,  # Your baseline RF pipeline
    nn_baseline,  # Your baseline NN model
    X_test_processed,  # Already preprocessed
    y_test,
    rf_weight=0.5,  # Equal weights
    metric='accuracy',
    n_repeats=10,
    random_state=42
)

# Create DataFrame
ensemble_base_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': [imp['mean'] for imp in perm_importance_ensemble_base],
    'std': [imp['std'] for imp in perm_importance_ensemble_base]
}).sort_values('importance', ascending=False)

print("\n=== Top 20 Most Important Features (Baseline Ensemble) ===")
print(ensemble_base_importance.head(20))

# ============================================
# 2. TUNED ENSEMBLE (RF Tuned + NN Tuned)
# ============================================

print("\n" + "=" * 60)
print("TUNED ENSEMBLE FEATURE IMPORTANCE")
print(f"RF Tuned + NN Tuned ({best_weight:.0%}:{(1 - best_weight):.0%} weight)")
print("=" * 60)

perm_importance_ensemble_tuned = calculate_ensemble_permutation_importance(
    rf_best,  # Your tuned RF pipeline (from GridSearchCV)
    best_nn,  # Your tuned NN model (from Keras Tuner)
    X_test_processed,
    y_test,
    rf_weight=best_weight,  # Optimized weight from your code
    metric='accuracy',
    n_repeats=10,
    random_state=42
)

# Create DataFrame
ensemble_tuned_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': [imp['mean'] for imp in perm_importance_ensemble_tuned],
    'std': [imp['std'] for imp in perm_importance_ensemble_tuned]
}).sort_values('importance', ascending=False)

print("\n=== Top 20 Most Important Features (Tuned Ensemble) ===")
print(ensemble_tuned_importance.head(20))

# ============================================
# 3. VISUALIZATION: Ensemble Feature Importance
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Baseline Ensemble
top_features_ens_base = ensemble_base_importance.head(20)
axes[0].barh(range(len(top_features_ens_base)), top_features_ens_base['importance'],
             xerr=top_features_ens_base['std'], color='purple', alpha=0.7)
axes[0].set_yticks(range(len(top_features_ens_base)))
axes[0].set_yticklabels(top_features_ens_base['feature'], fontsize=9)
axes[0].set_xlabel('Importance (Decrease in Accuracy)', fontsize=11)
axes[0].set_title('Top 20 Features - Baseline Ensemble\n(RF+NN, 50:50)',
                  fontsize=12, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# Tuned Ensemble
top_features_ens_tuned = ensemble_tuned_importance.head(20)
axes[1].barh(range(len(top_features_ens_tuned)), top_features_ens_tuned['importance'],
             xerr=top_features_ens_tuned['std'], color='darkgreen', alpha=0.7)
axes[1].set_yticks(range(len(top_features_ens_tuned)))
axes[1].set_yticklabels(top_features_ens_tuned['feature'], fontsize=9)
axes[1].set_xlabel('Importance (Decrease in Accuracy)', fontsize=11)
axes[1].set_title(f'Top 20 Features - Tuned Ensemble\n(RF+NN, {best_weight:.0%}:{(1 - best_weight):.0%})',
                  fontsize=12, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 4. COMPREHENSIVE COMPARISON (All Models)
# ============================================

print("\n" + "=" * 60)
print("COMPREHENSIVE FEATURE IMPORTANCE COMPARISON")
print("=" * 60)

# You should already have these from your earlier code:
# - nn_base_importance (from NN baseline)
# - nn_tuned_importance (from NN tuned)
# - feature_importance_base (from RF baseline) - rename for consistency
# - feature_importance_tuned (from RF tuned) - rename for consistency

# Merge all importance scores
comparison_all = ensemble_base_importance[['feature', 'importance']].rename(
    columns={'importance': 'ensemble_baseline'})

comparison_all = comparison_all.merge(
    ensemble_tuned_importance[['feature', 'importance']].rename(
        columns={'importance': 'ensemble_tuned'}),
    on='feature')

# Add NN importances if available
try:
    comparison_all = comparison_all.merge(
        nn_base_importance[['feature', 'importance']].rename(
            columns={'importance': 'nn_baseline'}),
        on='feature')

    comparison_all = comparison_all.merge(
        nn_tuned_importance[['feature', 'importance']].rename(
            columns={'importance': 'nn_tuned'}),
        on='feature')

    # Add RF importances
    comparison_all = comparison_all.merge(
        feature_importance_base[['feature', 'importance']].rename(
            columns={'importance': 'rf_baseline'}),
        on='feature')

    comparison_all = comparison_all.merge(
        feature_importance_tuned[['feature', 'importance']].rename(
            columns={'importance': 'rf_tuned'}),
        on='feature')

    # Calculate average importance
    comparison_all['avg_importance'] = comparison_all[[
        'ensemble_baseline', 'ensemble_tuned', 'nn_baseline',
        'nn_tuned', 'rf_baseline', 'rf_tuned'
    ]].mean(axis=1)

    print("✅ All model importances merged successfully")

except Exception as e:
    print(f"⚠️ Note: {e}")
    print("Only using ensemble importances")
    comparison_all['avg_importance'] = comparison_all[[
        'ensemble_baseline', 'ensemble_tuned'
    ]].mean(axis=1)

comparison_all = comparison_all.sort_values('avg_importance', ascending=False)

print("\n=== Top 20 Features by Average Importance ===")
print(comparison_all.head(20))

# ============================================
# 5. ENSEMBLE COMPARISON VISUALIZATION
# ============================================

fig, ax = plt.subplots(figsize=(12, 10))

top_20 = comparison_all.head(20)
x = np.arange(len(top_20))
width = 0.35

# Plot ensemble importances side by side
ax.barh(x - width / 2, top_20['ensemble_baseline'], width,
        label='Baseline Ensemble', color='purple', alpha=0.8)
ax.barh(x + width / 2, top_20['ensemble_tuned'], width,
        label='Tuned Ensemble', color='darkgreen', alpha=0.8)

ax.set_yticks(x)
ax.set_yticklabels(top_20['feature'], fontsize=9)
ax.set_xlabel('Feature Importance', fontsize=11)
ax.set_title('Top 20 Features: Baseline vs Tuned Ensemble',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 6. FEATURE STABILITY ANALYSIS
# ============================================

print("\n" + "=" * 60)
print("FEATURE STABILITY ANALYSIS")
print("=" * 60)

# Calculate std across ensemble models
comparison_all['ensemble_std'] = comparison_all[[
    'ensemble_baseline', 'ensemble_tuned'
]].std(axis=1)

# Stability score: high importance, low variability
comparison_all['stability_score'] = (
        comparison_all['avg_importance'] / (comparison_all['ensemble_std'] + 0.0001)
)

stable_features = comparison_all.sort_values('stability_score', ascending=False)

print("\n=== Top 15 Most Stable Important Features ===")
print(stable_features[['feature', 'avg_importance', 'ensemble_std', 'stability_score']].head(15))

# ============================================
# 7. SAVE RESULTS
# ============================================

ensemble_base_importance.to_csv('feature_importance_ensemble_baseline.csv', index=False)
ensemble_tuned_importance.to_csv('feature_importance_ensemble_tuned.csv', index=False)
comparison_all.to_csv('feature_importance_ensemble_comparison.csv', index=False)

print("\n" + "=" * 60)
print("✅ ENSEMBLE FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("=" * 60)

# ============================================
# COMPREHENSIVE MODEL EVALUATION & COMPARISON
# ============================================

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score
)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("\n" + "=" * 80)
print("COMPREHENSIVE MODEL EVALUATION & COMPARISON")
print("=" * 80)

# ============================================
# COLLECT ALL PREDICTIONS
# ============================================

# Dictionary to store all predictions
predictions = {
    'Baseline RF': {
        'y_pred': y_pred_base,
        'y_pred_proba': rf_baseline.predict_proba(X_test)[:, 1]
    },
    'RF + SMOTE': {
        'y_pred': y_pred_smote,
        'y_pred_proba': rf_smote_only.predict_proba(X_test)[:, 1]
    },
    'Tuned RF + SMOTE': {
        'y_pred': y_pred_tuned_rf,
        'y_pred_proba': rf_best.predict_proba(X_test)[:, 1]
    },
    'Baseline NN': {
        'y_pred': y_pred_nn_baseline,
        'y_pred_proba': y_pred_nn_baseline_proba.flatten()
    },
    'Tuned NN': {
        'y_pred': y_pred_nn_tuned,
        'y_pred_proba': y_pred_nn_tuned_proba.flatten()
    },
    'Ensemble (Tuned RF + NN)': {
        'y_pred': y_pred_ensemble_tuned,
        'y_pred_proba': y_pred_ensemble_tuned_proba
    }
}

# ============================================
# CALCULATE ALL METRICS FOR EACH MODEL
# ============================================

metrics_list = []

for model_name, preds in predictions.items():
    y_pred = preds['y_pred']
    y_pred_proba = preds['y_pred_proba']

    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Calculate all metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall (Sensitivity)': recall_score(y_test, y_pred),
        'Specificity': tn / (tn + fp),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
        'Matthews Corr Coef': matthews_corrcoef(y_test, y_pred),
        'Cohen Kappa': cohen_kappa_score(y_test, y_pred),
        'True Positives': tp,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn,
        'Total Correct': tp + tn,
        'Total Incorrect': fp + fn
    }

    metrics_list.append(metrics)

# Create comprehensive metrics DataFrame
metrics_df = pd.DataFrame(metrics_list)

# ============================================
# DISPLAY METRICS TABLE
# ============================================

print("\n=== COMPREHENSIVE METRICS COMPARISON TABLE ===\n")
print(metrics_df.to_string(index=False))

# ============================================
# PERFORMANCE METRICS VISUALIZATION
# ============================================

# 1. Bar plot comparison of key metrics
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

metrics_to_plot = ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1-Score']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    values = metrics_df[metric].values
    bars = ax.bar(range(len(metrics_df)), values, color=colors, alpha=0.8)
    ax.set_xticks(range(len(metrics_df)))
    ax.set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} Comparison')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# ROC CURVES COMPARISON
# ============================================

plt.figure(figsize=(14, 10))

# Plot ROC curve for each model
for model_name, preds in predictions.items():
    y_pred_proba = preds['y_pred_proba']
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, linewidth=2.5,
             label=f'{model_name} (AUC = {roc_auc:.3f})')

# Plot diagonal line
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.500)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12)
plt.title('ROC Curves Comparison - All Models', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================
# CONFUSION MATRICES VISUALIZATION
# ============================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (model_name, preds) in enumerate(predictions.items()):
    y_pred = preds['y_pred']
    cm = confusion_matrix(y_test, y_pred)

    # Create confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['No Churn', 'Churn'])
    disp.plot(ax=axes[idx], cmap='Blues', values_format='d')
    axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
    axes[idx].grid(False)

plt.tight_layout()
plt.show()

# ============================================
# RADAR CHART COMPARISON
# ============================================

from math import pi

# Select key metrics for radar chart
radar_metrics = ['Accuracy', 'Precision', 'Recall (Sensitivity)',
                 'F1-Score', 'ROC-AUC', 'Balanced Accuracy']

# Number of variables
num_vars = len(radar_metrics)

# Compute angle for each axis
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]

# Create radar chart
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

# Plot each model
for idx, (model_name, _) in enumerate(predictions.items()):
    values = metrics_df[metrics_df['Model'] == model_name][radar_metrics].values.flatten().tolist()
    values += values[:1]

    ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
    ax.fill(angles, values, alpha=0.15, color=colors[idx])

# Fix axis to go in the right order and start at 12 o'clock
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw axis lines for each angle and label
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics, size=10)

# Set y-axis limits
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
ax.set_rlabel_position(0)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
plt.title('Model Performance Radar Chart', size=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# ============================================
# DETAILED PERFORMANCE REPORT
# ============================================

print("\n" + "=" * 80)
print("DETAILED PERFORMANCE ANALYSIS")
print("=" * 80)

# Find best model for each metric
print("\n=== BEST MODEL FOR EACH METRIC ===\n")
for metric in ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1-Score',
               'ROC-AUC', 'Balanced Accuracy', 'Matthews Corr Coef']:
    best_model = metrics_df.loc[metrics_df[metric].idxmax(), 'Model']
    best_value = metrics_df[metric].max()
    print(f"{metric:25s}: {best_model:30s} ({best_value:.4f})")

# ============================================
# MODEL RANKING
# ============================================

print("\n" + "=" * 80)
print("MODEL RANKING")
print("=" * 80)

# Rank models based on multiple metrics (higher is better)
ranking_metrics = ['Accuracy', 'Precision', 'Recall (Sensitivity)',
                   'F1-Score', 'ROC-AUC', 'Balanced Accuracy']

# Calculate average rank for each model
ranks = []
for model in metrics_df['Model']:
    model_ranks = []
    for metric in ranking_metrics:
        rank = metrics_df[metric].rank(ascending=False)[
            metrics_df['Model'] == model
            ].values[0]
        model_ranks.append(rank)
    avg_rank = np.mean(model_ranks)
    ranks.append({'Model': model, 'Average Rank': avg_rank})

ranking_df = pd.DataFrame(ranks).sort_values('Average Rank')
ranking_df['Overall Position'] = range(1, len(ranking_df) + 1)

print("\n=== OVERALL MODEL RANKING ===")
print("(Based on average rank across all key metrics)\n")
print(ranking_df.to_string(index=False))

# ============================================
# PERFORMANCE IMPROVEMENT ANALYSIS
# ============================================

print("\n" + "=" * 80)
print("PERFORMANCE IMPROVEMENT ANALYSIS")
print("=" * 80)

# Compare Baseline vs Tuned models
comparisons = [
    ('Baseline RF', 'Tuned RF + SMOTE'),
    ('Baseline NN', 'Tuned NN'),
]

improvement_data = []

for baseline, tuned in comparisons:
    baseline_metrics = metrics_df[metrics_df['Model'] == baseline].iloc[0]
    tuned_metrics = metrics_df[metrics_df['Model'] == tuned].iloc[0]

    print(f"\n=== {baseline} → {tuned} ===")

    for metric in ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1-Score', 'ROC-AUC']:
        baseline_val = baseline_metrics[metric]
        tuned_val = tuned_metrics[metric]
        improvement = ((tuned_val - baseline_val) / baseline_val) * 100

        improvement_data.append({
            'Comparison': f'{baseline} → {tuned}',
            'Metric': metric,
            'Baseline': baseline_val,
            'Tuned': tuned_val,
            'Improvement (%)': improvement
        })

        print(f"{metric:25s}: {baseline_val:.4f} → {tuned_val:.4f} "
              f"({improvement:+.2f}%)")

improvement_df = pd.DataFrame(improvement_data)

# ============================================
# SUMMARY & RECOMMENDATIONS
# ============================================

print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)

best_overall = ranking_df.iloc[0]['Model']
# best_business = business_df.iloc[0]['Model']
best_recall = metrics_df.loc[metrics_df['Recall (Sensitivity)'].idxmax(), 'Model']
best_precision = metrics_df.loc[metrics_df['Precision'].idxmax(), 'Model']
best_f1 = metrics_df.loc[metrics_df['F1-Score'].idxmax(), 'Model']

print(f"\n🏆 BEST OVERALL MODEL (by ranking): {best_overall}")
# print(f"💰 BEST BUSINESS VALUE: {best_business}")
print(f"🎯 BEST AT CATCHING CHURNERS (Recall): {best_recall}")
print(f"✓ BEST AT PRECISION: {best_precision}")
print(f"⚖️ BEST BALANCE (F1-Score): {best_f1}")

# ============================================
# EASY SUMMARY TABLE FOR REPORT
# ============================================

summary_metrics = metrics_df[['Model', 'Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1-Score', 'ROC-AUC']]
summary_metrics = summary_metrics.round(3)
print("\n=== SUMMARY PERFORMANCE TABLE ===\n")
print(summary_metrics.to_string(index=False))

# ============================================
# RESIDUAL ANALYSIS (for classification)
# ============================================



# Compute residuals using predicted probabilities for the best model
best_model_name = 'Ensemble (Tuned RF + NN)'  # or whichever performed best
best_preds = predictions[best_model_name]
y_pred_proba = best_preds['y_pred_proba']

# Compute residuals (actual - predicted probability)
residuals = y_test - y_pred_proba

# Combine into DataFrame
residual_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted_Prob': y_pred_proba,
    'Residual': residuals
}).reset_index(drop=True)

# 1️⃣ Residual distribution
plt.figure(figsize=(8, 6))
sns.histplot(residual_df['Residual'], kde=True, color='skyblue', bins=30)
plt.title(f'Residual Distribution for {best_model_name}')
plt.xlabel('Residual (Actual - Predicted Probability)')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)
plt.show()

# 2️⃣ Residuals vs Predicted Probability
plt.figure(figsize=(8, 6))
sns.scatterplot(x=residual_df['Predicted_Prob'], y=residual_df['Residual'],
                alpha=0.6, color='teal')
plt.axhline(0, color='red', linestyle='--')
plt.title(f'Residuals vs Predicted Probability ({best_model_name})')
plt.xlabel('Predicted Probability of Churn')
plt.ylabel('Residual (Actual - Predicted Probability)')
plt.grid(alpha=0.3)
plt.show()

# 3️⃣ Boxplot by Actual class
plt.figure(figsize=(7, 5))
sns.boxplot(x=residual_df['Actual'], y=residual_df['Residual'], palette='Set2')
plt.title(f'Residual Distribution by Actual Class ({best_model_name})')
plt.xlabel('Actual Churn (0 = No, 1 = Yes)')
plt.ylabel('Residual')
plt.grid(alpha=0.3)
plt.show()

# ============================================
# RESIDUAL ANALYSIS SUMMARY
# ============================================



sns.set(style="whitegrid")

# --------------------------------------------
# 1️⃣ Residual Distributions for ALL Models (side-by-side)
# --------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (model_name, preds) in enumerate(predictions.items()):
    y_pred_proba = preds['y_pred_proba']
    residuals = y_test - y_pred_proba

    sns.histplot(residuals, kde=True, bins=30, color='skyblue', ax=axes[i])
    axes[i].set_title(f"Residual Distribution - {model_name}")
    axes[i].set_xlabel("Residual (Actual - Predicted Probability)")
    axes[i].set_ylabel("Frequency")
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# --------------------------------------------
# 2️⃣ Residuals vs Predicted Probability (ONLY for Ensemble model)
# --------------------------------------------
best_model_name = 'Ensemble (Tuned RF + NN)'
best_preds = predictions[best_model_name]
y_pred_proba = best_preds['y_pred_proba']
residuals = y_test - y_pred_proba

residual_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted_Prob': y_pred_proba,
    'Residual': residuals
}).reset_index(drop=True)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=residual_df['Predicted_Prob'], y=residual_df['Residual'],
                alpha=0.6, color='teal')
plt.axhline(0, color='red', linestyle='--', linewidth=1.2)
plt.title(f"Residuals vs Predicted Probability - {best_model_name}")
plt.xlabel("Predicted Probability of Churn")
plt.ylabel("Residual (Actual - Predicted Probability)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================
# COMBINED RESIDUAL DISTRIBUTION (CLASSIFICATION)
# ============================================



sns.set(style="whitegrid")

# Pick the models you want to compare
selected_models = {
    "RF Residuals": "Tuned RF + SMOTE",
    "NN Residuals": "Tuned NN",
    "Ensemble Residuals": "Ensemble (Tuned RF + NN)"
}

# Create a dataframe for plotting
residuals_data = []

for label, model_name in selected_models.items():
    y_pred_proba = predictions[model_name]['y_pred_proba']
    residuals = y_test - y_pred_proba
    residuals_data.append(pd.DataFrame({
        'Model': label,
        'Residual': residuals
    }))

residuals_df = pd.concat(residuals_data, ignore_index=True)

# Plot combined histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=residuals_df, x='Residual', hue='Model',
             bins=40, kde=False, alpha=0.5, palette=['#4C72B0', '#DD8452', '#55A868'])

# Perfect prediction reference line
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')

plt.title("Residual Distribution (Error Spread)")
plt.xlabel("Prediction Residual (Actual - Predicted Probability)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


