#!/usr/bin/env python
# coding: utf-8

# # Comparative Analysis of Machine Learning and Bio-Inspired Optimization Algorithms for Road Accident Severity Prediction
# 
# ## Phase 1: Dataset Loading and Detailed Inspection
# In this section, we load the CSV dataset, check its dimensions, display column names and data types, check for missing values, analyze potential target variables (accident severity columns), and identify potential target leakage columns.

# In[1]:


import pandas as pd
import numpy as np

csv_path = r"C:\Users\athis\Downloads\5df1f373-0c90-48f5-80e1-7b2a35507134.csv"
print(f"Loading dataset from: {csv_path}")
df = pd.read_csv(csv_path, low_memory=False)

print("\n--- 1. Shape of Dataset ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- 2. Columns & Data Types ---")
display(df.dtypes.to_frame(name='DataType'))

print("\n--- 3. First 10 Rows ---")
display(df.head(10))

print("\n--- 4. Missing Values Summary ---")
null_summary = df.isnull().sum().to_frame(name='MissingCount')
null_summary['Percentage (%)'] = (null_summary['MissingCount'] / len(df)) * 100
display(null_summary[null_summary['MissingCount'] > 0].sort_values(by='MissingCount', ascending=False))

print("\n--- 5. Duplicate Records ---")
duplicates = df.duplicated().sum()
print(f"Number of exact duplicate rows: {duplicates}")

# Identify possible severity columns
severity_candidates = [col for col in df.columns if 'severity' in col.lower() or 'sev' in col.lower()]
print(f"\n--- 6. Potential Severity Columns Detected: {severity_candidates} ---")
for col in severity_candidates:
    print(f"\nUnique values and class distribution for: '{col}'")
    display(df[col].value_counts(dropna=False).to_frame(name='Count'))

# Identify columns representing counts or identifiers (Potential Leakage)
leakage_keywords = ['injur', 'kill', 'fatal', 'death', 'casualty', 'no_persons', 'accident_no', 'accident_num']
leakage_candidates = [col for col in df.columns if any(kw in col.lower() for kw in leakage_keywords)]
print(f"\n--- 7. Potential Target Leakage Columns: ---\n{leakage_candidates}")


# ### Phase 1 (Continued): Target Class Distribution, Target Mapping, and Leakage Identification
# We will examine the exact labels in the `SEVERITY` column, perform the proposed mapping, verify the distribution of the mapped target variable, and finalize our plan for target leakage prevention.

# In[2]:


import pandas as pd
import numpy as np

# Load dataset
csv_path = r"C:\Users\athis\Downloads\5df1f373-0c90-48f5-80e1-7b2a35507134.csv"
df = pd.read_csv(csv_path, low_memory=False)

print("--- Unique values in original SEVERITY column ---")
print(df['SEVERITY'].value_counts(dropna=False))

# Map severity to 3 classes (0: Non-injury, 1: Injury, 2: Fatal)
def map_severity(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower()
    if 'fatal' in val_str:
        return 2
    elif 'serious' in val_str or 'injury' in val_str or 'other' in val_str:
        return 1
    elif 'non' in val_str or 'no injury' in val_str or 'non injury' in val_str:
        return 0
    else:
        if '1' in val_str:
            return 2 # Fatal
        elif '2' in val_str or '3' in val_str:
            return 1 # Serious/Other Injury
        elif '4' in val_str:
            return 0 # Non-injury
        return np.nan

df['target'] = df['SEVERITY'].apply(map_severity)

print("\n--- Target Mapping Distribution ---")
print(df['target'].value_counts(dropna=False))

print("\n--- Verifying Leakage and Identifiers to Exclude ---")
leakage_cols = [
    'ACCIDENT_NO', '_id', 'FATALITY', 'SERIOUSINJURY',
    'OTHERINJURY', 'NONINJURED', 'INJ_OR_FATAL', 'TOTAL_PERSONS'
]
print("Columns identified as target leakage or unique IDs to be excluded:")
for col in leakage_cols:
    if col in df.columns:
        print(f" - {col} (Exists in dataset)")


# ## Phase 2: Data Cleaning, Preprocessing Pipeline, and Feature Engineering
# In this section, we will:
# 1. Re-map the target variable to binary classification to resolve the 4-sample class issue:
#    - Class 0: Non-fatal (comprising Non-injury, Other injury, and Serious injury) -> **198,915 samples**
#    - Class 1: Fatal -> **3,386 samples**
# 2. Prevent data leakage by dropping identifiers, counts (`FATALITY`, `SERIOUSINJURY`, `OTHERINJURY`, `NONINJURED`, `INJ_OR_FATAL`, `TOTAL_PERSONS`, `ACCIDENT_NO`, `_id`), and the original `SEVERITY` column.
# 3. Clean features: handle missing values, decode categorical columns, process datetimes (`ACCIDENT_DATE`, `ACCIDENT_TIME`), and prepare numeric columns.
# 4. Apply Stratified Sampling for scalable performance, and perform train/validation/test splits.

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Define target mapping: 0 for Non-fatal (Injury/Non-injury), 1 for Fatal
def binary_target_map(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower()
    if 'fatal' in val_str:
        return 1
    return 0

df['target'] = df['SEVERITY'].apply(binary_target_map)
print("--- Binary Target Distribution ---")
print(df['target'].value_counts(dropna=False))

# 2. Prevent Data Leakage: Explicitly drop columns revealing outcomes or counts
leakage_columns = [
    '_id', 'ACCIDENT_NO', 'SEVERITY', 'FATALITY', 'SERIOUSINJURY',
    'OTHERINJURY', 'NONINJURED', 'INJ_OR_FATAL', 'TOTAL_PERSONS'
]

# Dropping columns that are highly correlated with vehicle/person level outcomes
person_vehicle_outcomes = [
    'MALES', 'FEMALES', 'BICYCLIST', 'PASSENGER', 'DRIVER', 'PEDESTRIAN',
    'PILLION', 'MOTORCYCLIST', 'UNKNOWN', 'PED_CYCLIST_5_12',
    'PED_CYCLIST_13_18', 'OLD_PED_65_AND_OVER', 'OLD_DRIVER_75_AND_OVER',
    'YOUNG_DRIVER_18_25'
]

columns_to_drop = leakage_columns + person_vehicle_outcomes
cleaned_df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# 3. Handle Datetime Parsing
if 'ACCIDENT_DATE' in cleaned_df.columns:
    cleaned_df['ACCIDENT_DATE'] = pd.to_datetime(cleaned_df['ACCIDENT_DATE'], errors='coerce')
    cleaned_df['Month'] = cleaned_df['ACCIDENT_DATE'].dt.month
    cleaned_df['Day'] = cleaned_df['ACCIDENT_DATE'].dt.day
    cleaned_df.drop(columns=['ACCIDENT_DATE'], inplace=True)

if 'ACCIDENT_TIME' in cleaned_df.columns:
    cleaned_df['Hour'] = pd.to_datetime(cleaned_df['ACCIDENT_TIME'], format='%H:%M:%S', errors='coerce').dt.hour
    idx_null = cleaned_df['Hour'].isnull()
    if idx_null.any():
        cleaned_df.loc[idx_null, 'Hour'] = pd.to_datetime(cleaned_df.loc[idx_null, 'ACCIDENT_TIME'], errors='coerce').dt.hour
    cleaned_df.drop(columns=['ACCIDENT_TIME'], inplace=True)

print(f"\nShape after leakage exclusion and datetime feature engineering: {cleaned_df.shape}")


# ### Phase 2 (Continued): Preprocessing Pipeline, Stratified Splits & Class Distribution Verification
# In this section, we build a robust preprocessing routine for missing values, handle categorical columns, split the dataset into 64% Train, 16% Validation, and 20% Test sets, and verify that the target classes are properly stratified across all splits.

# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Separate features and target
X = cleaned_df.drop(columns=['target'])
y = cleaned_df['target']

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Numerical columns ({len(num_cols)}): {num_cols}")
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

# Impute missing values securely without target leakage
for col in num_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)

for col in cat_cols:
    X[col] = X[col].fillna('Unknown').astype(str)

print("\n--- Missing Values after Imputation ---")
print(X.isnull().sum().sum())

# Perform reproducible stratified splits: 80% (train + val), 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Split remaining 80% into 80% Train and 20% Validation (16% overall)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.20, random_state=42, stratify=y_train_val
)

print("\n--- Split Class Distributions ---")
for name, target_series in zip(['Train', 'Validation', 'Test'], [y_train, y_val, y_test]):
    counts = target_series.value_counts()
    ratios = target_series.value_counts(normalize=True) * 100
    print(f"\n{name} Split (Total: {len(target_series)}):")
    for cls in [0, 1]:
        print(f"  Class {cls}: {counts.get(cls, 0)} ({ratios.get(cls, 0.0):.2f}%)")


# ## Phase 3: Baseline machine-learning models
# 
# We retain the existing **binary target**: `0 = Non-fatal` and `1 = Fatal`. The following baseline models are benchmarked:
# 1. Random Forest
# 2. XGBoost
# 3. LightGBM
# 4. CatBoost

# In[5]:


get_ipython().run_line_magic('pip', '-q install catboost xgboost lightgbm imbalanced-learn shap')


# In[6]:


import time, warnings, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style='whitegrid')

# Verify split partitions
for name, obj in [('X_train', X_train), ('X_val', X_val), ('X_test', X_test),
                  ('y_train', y_train), ('y_val', y_val), ('y_test', y_test)]:
    print(f'{name}: {obj.shape}')

assert len(X_train) > 0 and len(X_val) > 0 and len(X_test) > 0, 'A data split is empty.'
assert set(pd.Series(y_train).unique()) == {0, 1}, 'Both target classes must be present in training data.'

X_train = X_train.copy(); X_val = X_val.copy(); X_test = X_test.copy()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
for frame in (X_train, X_val, X_test):
    for col in cat_cols:
        frame[col] = frame[col].fillna('Unknown').astype(str)
    for col in num_cols:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')

fatal_count = int((y_train == 1).sum())
nonfatal_count = int((y_train == 0).sum())
scale_pos_weight = nonfatal_count / fatal_count
print(f'Fatal class weight ratio: {scale_pos_weight:.2f}')
print(f'Numerical features: {len(num_cols)} | Categorical features: {len(cat_cols)}')


# ### 3.1 Common evaluation functions and preprocessing

# In[7]:


try:
    onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=True, min_frequency=2)
except TypeError:
    onehot = OneHotEncoder(handle_unknown='ignore', sparse=True)

numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', onehot)
])
preprocessor = ColumnTransformer([
    ('num', numeric_pipe, num_cols),
    ('cat', categorical_pipe, cat_cols)
], remainder='drop')

X_train_enc = preprocessor.fit_transform(X_train)
X_val_enc = preprocessor.transform(X_val)
X_test_enc = preprocessor.transform(X_test)
print('Encoded matrix shapes:', X_train_enc.shape, X_val_enc.shape, X_test_enc.shape)

baseline_results = {}

def calculate_metrics(y_true, y_pred, y_prob=None):
    result = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Macro F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Weighted F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'Fatal Precision': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Fatal Recall': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Fatal F1': f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        result['ROC AUC'] = roc_auc_score(y_true, y_prob)
        result['PR AUC'] = average_precision_score(y_true, y_prob)
    return result

def record_validation(name, model, Xv, yv, train_seconds):
    pred = np.asarray(model.predict(Xv)).reshape(-1).astype(int)
    prob = model.predict_proba(Xv)[:, 1] if hasattr(model, 'predict_proba') else None
    metrics = calculate_metrics(yv, pred, prob)
    metrics['Training Time (s)'] = train_seconds
    baseline_results[name] = metrics
    print(f'\n{name} — validation metrics')
    display(pd.Series(metrics).to_frame('Value'))
    print(classification_report(yv, pred, labels=[0, 1], target_names=['Non-fatal', 'Fatal'], zero_division=0))
    return pred, prob


# ### 3.2 Train the four baseline models

# In[8]:


# Random Forest Baseline
rf_model = RandomForestClassifier(
    n_estimators=300, class_weight='balanced_subsample',
    min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
)
t0 = time.time(); rf_model.fit(X_train_enc, y_train); rf_time = time.time() - t0
pred_rf, prob_rf = record_validation('Random Forest', rf_model, X_val_enc, y_val, rf_time)

# XGBoost Baseline
xgb_model = XGBClassifier(
    n_estimators=350, max_depth=6, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85,
    scale_pos_weight=scale_pos_weight, objective='binary:logistic',
    eval_metric='aucpr', tree_method='hist', random_state=RANDOM_STATE,
    n_jobs=-1
)
t0 = time.time(); xgb_model.fit(X_train_enc, y_train); xgb_time = time.time() - t0
pred_xgb, prob_xgb = record_validation('XGBoost', xgb_model, X_val_enc, y_val, xgb_time)

# LightGBM Baseline
lgb_model = LGBMClassifier(
    n_estimators=400, learning_rate=0.05, num_leaves=31,
    class_weight='balanced', random_state=RANDOM_STATE,
    n_jobs=-1, verbosity=-1
)
t0 = time.time(); lgb_model.fit(X_train_enc, y_train); lgb_time = time.time() - t0
pred_lgb, prob_lgb = record_validation('LightGBM', lgb_model, X_val_enc, y_val, lgb_time)

# CatBoost Baseline
cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]
cb_model = CatBoostClassifier(
    iterations=500, depth=7, learning_rate=0.05,
    auto_class_weights='Balanced', loss_function='Logloss',
    eval_metric='AUC', random_seed=RANDOM_STATE,
    verbose=False, allow_writing_files=False, thread_count=-1
)
t0 = time.time()
cb_model.fit(X_train, y_train, cat_features=cat_indices, eval_set=(X_val, y_val),
             early_stopping_rounds=60, verbose=False)
cb_time = time.time() - t0
pred_cb, prob_cb = record_validation('CatBoost', cb_model, X_val, y_val, cb_time)

baseline_table = pd.DataFrame(baseline_results).T.sort_values('Macro F1', ascending=False)
display(baseline_table.style.format('{:.4f}'))


# ## Phase 4: Bio-inspired feature selection — ACO and BPSO (Evaluated with Random Forest)
# 
# ACO and BPSO select from the **original input columns**. We evaluate feature sub-masks using **Random Forest** (with `class_weight='balanced'`) as the internal wrapper evaluator rather than LightGBM.
# 
# The fitness objective is the validation macro F1 score minus a feature size penalty.

# In[9]:


search_preprocessor = preprocessor # use the same fitted ColumnTransformer

# Use stratified samples for manageable optimization runtime
def stratified_sample(Xpart, ypart, max_n, seed):
    if len(Xpart) <= max_n:
        return Xpart.copy(), ypart.copy()
    frac = max_n / len(Xpart)
    Xs, _, ys, _ = train_test_split(
        Xpart, ypart, train_size=frac, random_state=seed, stratify=ypart
    )
    return Xs, ys

Xtr_s, ytr_s = stratified_sample(X_train, y_train, 12000, RANDOM_STATE)
Xva_s, yva_s = stratified_sample(X_val, y_val, 4000, RANDOM_STATE + 1)
Xtr_s_enc = preprocessor.transform(Xtr_s)
Xva_s_enc = preprocessor.transform(Xva_s)

# Original input feature names
search_feature_names = num_cols + cat_cols
num_search_features = len(search_feature_names)
print('ACO/BPSO search shapes:', Xtr_s_enc.shape, Xva_s_enc.shape)
print('Search features:', num_search_features)

# Helper function to map a binary feature mask of original features to one-hot feature matrix columns
def get_encoded_column_mask(feature_mask):
    try:
        cat_encoded_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
    except AttributeError:
        cat_encoded_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names(cat_cols)

    encoded_cols = num_cols + list(cat_encoded_names)

    cols_to_keep = []
    selected_orig_features = [f for f, keep in zip(search_feature_names, feature_mask) if keep]

    for idx, col in enumerate(encoded_cols):
        if any(col == f or col.startswith(f + '_') for f in selected_orig_features):
            cols_to_keep.append(idx)
    return cols_to_keep

# Random Forest based feature mask evaluation function
def evaluate_feature_mask(mask, n_estimators=100):
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return 0.0

    col_indices = get_encoded_column_mask(mask)
    if len(col_indices) == 0:
        return 0.0

    rf_eval = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_eval.fit(Xtr_s_enc[:, col_indices], ytr_s)
    pred = rf_eval.predict(Xva_s_enc[:, col_indices])
    score = f1_score(yva_s, pred, average='macro', zero_division=0)
    size_penalty = 0.005 * (mask.sum() / len(mask))
    return score - size_penalty


# ### 4.1 Ant Colony Optimization (ACO) with Random Forest

# In[10]:


rng = np.random.default_rng(RANDOM_STATE)
N_ANTS, ACO_ITERATIONS = 8, 8
EVAPORATION, ALPHA, BETA = 0.25, 1.0, 1.0
pheromone = np.ones(num_search_features, dtype=float)
heuristic = np.ones(num_search_features, dtype=float)
aco_history = []
best_aco_mask, best_aco_fitness = None, -np.inf

for iteration in range(ACO_ITERATIONS):
    iteration_best_mask, iteration_best_score = None, -np.inf
    for ant in range(N_ANTS):
        probs = (pheromone ** ALPHA) * (heuristic ** BETA)
        probs = probs / probs.sum()
        mask = rng.random(num_search_features) < np.clip(probs, 0.15, 0.85)
        if mask.sum() == 0:
            mask[rng.integers(num_search_features)] = True
        fitness = evaluate_feature_mask(mask)
        if fitness > iteration_best_score:
            iteration_best_mask, iteration_best_score = mask.copy(), fitness
        if fitness > best_aco_fitness:
            best_aco_mask, best_aco_fitness = mask.copy(), fitness
    pheromone *= (1 - EVAPORATION)
    pheromone += iteration_best_mask.astype(float) * max(iteration_best_score, 0.01)
    aco_history.append(best_aco_fitness)
    print(f'ACO {iteration+1}/{ACO_ITERATIONS}: best fitness={best_aco_fitness:.4f}; features={best_aco_mask.sum()}')

aco_features = [f for f, keep in zip(search_feature_names, best_aco_mask) if keep]
print('ACO selected features:', aco_features)


# ### 4.2 Binary Particle Swarm Optimization (BPSO) with Random Forest

# In[11]:


rng = np.random.default_rng(RANDOM_STATE + 2)
N_PARTICLES, BPSO_ITERATIONS = 8, 8
W, C1, C2 = 0.72, 1.45, 1.45
positions = rng.integers(0, 2, size=(N_PARTICLES, num_search_features)).astype(bool)
velocities = rng.normal(0, 0.5, size=(N_PARTICLES, num_search_features))
pbest_positions = positions.copy()
pbest_scores = np.array([evaluate_feature_mask(p) for p in positions])
gbest_idx = int(np.argmax(pbest_scores))
gbest_position = pbest_positions[gbest_idx].copy()
gbest_score = float(pbest_scores[gbest_idx])
bpso_history = []

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -8, 8)))

for iteration in range(BPSO_ITERATIONS):
    for i in range(N_PARTICLES):
        if positions[i].sum() == 0:
            positions[i, rng.integers(num_search_features)] = True
        score = evaluate_feature_mask(positions[i])
        if score > pbest_scores[i]:
            pbest_scores[i] = score
            pbest_positions[i] = positions[i].copy()
        if score > gbest_score:
            gbest_score = score
            gbest_position = positions[i].copy()
    r1 = rng.random((N_PARTICLES, num_search_features))
    r2 = rng.random((N_PARTICLES, num_search_features))
    velocities = (W * velocities
                  + C1 * r1 * (pbest_positions.astype(float) - positions.astype(float))
                  + C2 * r2 * (gbest_position.astype(float) - positions.astype(float)))
    positions = rng.random((N_PARTICLES, num_search_features)) < sigmoid(velocities)
    bpso_history.append(gbest_score)
    print(f'BPSO {iteration+1}/{BPSO_ITERATIONS}: best fitness={gbest_score:.4f}; features={gbest_position.sum()}')

bpso_features = [f for f, keep in zip(search_feature_names, gbest_position) if keep]
print('BPSO selected features:', bpso_features)


# ## Phase 5: Genetic Algorithm for Random Forest hyperparameter optimization
# 
# The GA optimizes **Random Forest** hyperparameters on the BPSO-selected feature subset. Its fitness is validation macro F1. The hyperparameter search space includes `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`.

# In[12]:


if len(bpso_features) == 0:
    raise ValueError('BPSO returned an empty feature set.')

bpso_mask = np.isin(search_feature_names, bpso_features)
bpso_col_indices = get_encoded_column_mask(bpso_mask)

Xtr_ga = Xtr_s_enc[:, bpso_col_indices]
Xva_ga = Xva_s_enc[:, bpso_col_indices]

GA_POPULATION, GA_GENERATIONS = 8, 5
MUTATION_RATE, ELITE_COUNT = 0.25, 2
rng = np.random.default_rng(RANDOM_STATE + 3)

def random_individual():
    return {
        'n_estimators': int(rng.integers(100, 401)),
        'max_depth': int(rng.integers(5, 25)),
        'min_samples_split': int(rng.integers(2, 15)),
        'min_samples_leaf': int(rng.integers(1, 10)),
        'max_features': float(rng.uniform(0.3, 1.0))
    }

def ga_fitness(ind):
    model = RandomForestClassifier(
        **ind,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(Xtr_ga, ytr_s)
    pred = model.predict(Xva_ga)
    return f1_score(yva_s, pred, average='macro', zero_division=0)

def mutate(ind):
    child = ind.copy()
    if rng.random() < MUTATION_RATE: child['n_estimators'] = int(rng.integers(100, 401))
    if rng.random() < MUTATION_RATE: child['max_depth'] = int(rng.integers(5, 25))
    if rng.random() < MUTATION_RATE: child['min_samples_split'] = int(rng.integers(2, 15))
    if rng.random() < MUTATION_RATE: child['min_samples_leaf'] = int(rng.integers(1, 10))
    if rng.random() < MUTATION_RATE: child['max_features'] = float(rng.uniform(0.3, 1.0))
    return child

population = [random_individual() for _ in range(GA_POPULATION)]
ga_history = []
best_ga_params, best_ga_score = None, -np.inf
for generation in range(GA_GENERATIONS):
    scored = [(ga_fitness(ind), ind) for ind in population]
    scored.sort(key=lambda x: x[0], reverse=True)
    if scored[0][0] > best_ga_score:
        best_ga_score, best_ga_params = scored[0][0], scored[0][1].copy()
    ga_history.append(best_ga_score)
    print(f'GA generation {generation+1}/{GA_GENERATIONS}: best macro F1={best_ga_score:.4f}')
    next_population = [ind.copy() for _, ind in scored[:ELITE_COUNT]]
    while len(next_population) < GA_POPULATION:
        parent_pool = [ind for _, ind in scored[:max(3, GA_POPULATION // 2)]]
        p1, p2 = parent_pool[rng.integers(len(parent_pool))], parent_pool[rng.integers(len(parent_pool))]
        child = {k: (p1[k] if rng.random() < 0.5 else p2[k]) for k in p1}
        next_population.append(mutate(child))
    population = next_population

print('Best GA validation macro F1:', best_ga_score)
print('Best GA parameters:', best_ga_params)


# ## Phase 6: Train optimized Random Forest models and compare on validation data
# 
# We train Random Forest models using ACO-selected features, BPSO-selected features, and the GA-optimized Random Forest model on the BPSO feature subset.

# In[13]:


def get_feature_subset_matrix(matrix_enc, feature_list):
    mask = np.isin(search_feature_names, feature_list)
    indices = get_encoded_column_mask(mask)
    return matrix_enc[:, indices]

optimized_models = {}
optimized_validation = {}

for model_name, selected in [('ACO + Random Forest', aco_features), ('BPSO + Random Forest', bpso_features)]:
    model = RandomForestClassifier(
        n_estimators=300, class_weight='balanced_subsample',
        min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
    )
    Xt = get_feature_subset_matrix(X_train_enc, selected)
    Xv = get_feature_subset_matrix(X_val_enc, selected)
    t0 = time.time(); model.fit(Xt, y_train); elapsed = time.time() - t0
    pred = model.predict(Xv); prob = model.predict_proba(Xv)[:, 1]
    metrics = calculate_metrics(y_val, pred, prob); metrics['Training Time (s)'] = elapsed
    optimized_models[model_name] = (model, selected)
    optimized_validation[model_name] = metrics

# GA-optimized Random Forest trained on BPSO features
ga_rf_model = RandomForestClassifier(
    **best_ga_params, class_weight='balanced_subsample',
    random_state=RANDOM_STATE, n_jobs=-1
)
Xt = get_feature_subset_matrix(X_train_enc, bpso_features)
Xv = get_feature_subset_matrix(X_val_enc, bpso_features)
t0 = time.time(); ga_rf_model.fit(Xt, y_train); ga_time = time.time() - t0
pred = ga_rf_model.predict(Xv); prob = ga_rf_model.predict_proba(Xv)[:, 1]
metrics = calculate_metrics(y_val, pred, prob); metrics['Training Time (s)'] = ga_time
optimized_models['GA + BPSO + Random Forest'] = (ga_rf_model, bpso_features)
optimized_validation['GA + BPSO + Random Forest'] = metrics

all_validation = pd.concat([
    pd.DataFrame(baseline_results).T,
    pd.DataFrame(optimized_validation).T
], axis=0)
print('Validation model comparison:')
display(all_validation.sort_values('Macro F1', ascending=False).style.format('{:.4f}'))


# ## Phase 7: Final held-out test evaluation

# In[14]:


test_results = {}
test_predictions = {}

baseline_models = {
    'Random Forest': (rf_model, X_test_enc),
    'XGBoost': (xgb_model, X_test_enc),
    'LightGBM': (lgb_model, X_test_enc),
    'CatBoost': (cb_model, X_test)
}

for name, (model, Xte) in baseline_models.items():
    pred = np.asarray(model.predict(Xte)).reshape(-1).astype(int)
    prob = model.predict_proba(Xte)[:, 1]
    test_results[name] = calculate_metrics(y_test, pred, prob)
    test_predictions[name] = (pred, prob)

for name, (model, selected) in optimized_models.items():
    Xte = get_feature_subset_matrix(X_test_enc, selected)
    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)[:, 1]
    test_results[name] = calculate_metrics(y_test, pred, prob)
    test_predictions[name] = (pred, prob)

test_comparison = pd.DataFrame(test_results).T.sort_values('Macro F1', ascending=False)
print('Final test-set comparison:')
display(test_comparison.style.format('{:.4f}'))

best_validation_name = all_validation['Macro F1'].astype(float).idxmax()
print('Model selected using validation macro F1:', best_validation_name)


# ## Phase 8: Confusion matrices, classification reports and research visualizations

# In[15]:


for name, (pred, prob) in test_predictions.items():
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    plt.figure(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-fatal', 'Fatal'], yticklabels=['Non-fatal', 'Fatal'])
    plt.title(f'{name} — Test Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.tight_layout(); plt.show()
    print(f'\n{name} — Test classification report')
    print(classification_report(y_test, pred, labels=[0, 1],
                                target_names=['Non-fatal', 'Fatal'], zero_division=0))


# In[16]:


fig_count = 0

def show_plot(title):
    global fig_count
    fig_count += 1
    plt.title(f'{fig_count}. {title}')
    plt.tight_layout()
    plt.show()

# 1. Target distribution
plt.figure(figsize=(6, 4)); sns.countplot(x=pd.Series(y).map({0:'Non-fatal',1:'Fatal'}));
plt.xlabel('Crash outcome'); plt.ylabel('Number of records'); show_plot('Target class distribution')

# 2. Missingness in original source dataframe
missing_pct = (df.isna().mean() * 100).sort_values(ascending=False).head(20)
plt.figure(figsize=(9, 5)); missing_pct.sort_values().plot(kind='barh');
plt.xlabel('Missing values (%)'); show_plot('Top 20 columns by missingness')

# 3. Numeric feature distributions
for col in num_cols[:6]:
    plt.figure(figsize=(6, 4)); sns.histplot(data=X_train, x=col, hue=y_train, bins=30, element='step', stat='density', common_norm=False);
    plt.xlabel(col); show_plot(f'Distribution of {col} by target')

# 4. Categorical feature distributions
for col in cat_cols[:4]:
    top = X_train[col].value_counts().head(12).index
    plot_df = pd.DataFrame({col: X_train[col].where(X_train[col].isin(top), 'Other'), 'Target': y_train.map({0:'Non-fatal',1:'Fatal'})})
    plt.figure(figsize=(9, 4)); sns.countplot(data=plot_df, x=col, hue='Target', order=plot_df[col].value_counts().index);
    plt.xticks(rotation=45, ha='right'); show_plot(f'{col} category counts by target')

# 5. ACO convergence
plt.figure(figsize=(6, 4)); plt.plot(range(1,len(aco_history)+1), aco_history, marker='o');
plt.xlabel('Iteration'); plt.ylabel('Best validation fitness'); show_plot('ACO convergence')

# 6. BPSO convergence
plt.figure(figsize=(6, 4)); plt.plot(range(1,len(bpso_history)+1), bpso_history, marker='s');
plt.xlabel('Iteration'); plt.ylabel('Best validation fitness'); show_plot('BPSO convergence')

# 7. GA convergence
plt.figure(figsize=(6, 4)); plt.plot(range(1,len(ga_history)+1), ga_history, marker='^');
plt.xlabel('Generation'); plt.ylabel('Best validation macro F1'); show_plot('GA convergence')

# 8-11. Test model macro F1, fatal recall, accuracy and balanced accuracy
for metric in ['Macro F1','Fatal Recall','Accuracy','Balanced Accuracy']:
    plt.figure(figsize=(9, 4)); test_comparison[metric].sort_values().plot(kind='barh');
    plt.xlabel(metric); show_plot(f'Test-set model comparison — {metric}')

# 12. Test ROC curves
plt.figure(figsize=(7, 5))
for name, (pred, prob) in test_predictions.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, prob):.3f})')
plt.plot([0,1],[0,1],'k--'); plt.xlabel('False positive rate'); plt.ylabel('True positive rate');
plt.legend(fontsize=8); show_plot('Test ROC curves')

# 13. Precision-recall curves
plt.figure(figsize=(7, 5))
for name, (pred, prob) in test_predictions.items():
    precision, recall, _ = precision_recall_curve(y_test, prob)
    plt.plot(recall, precision, label=f'{name} (AP={average_precision_score(y_test, prob):.3f})')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(fontsize=8); show_plot('Test precision-recall curves')

# 14. Feature-selection size comparison
plt.figure(figsize=(6, 4));
plt.bar(['All features','ACO','BPSO'], [len(search_feature_names),len(aco_features),len(bpso_features)]);
plt.ylabel('Number of original features'); show_plot('Feature reduction comparison')

# 15. GA hyperparameter values
plt.figure(figsize=(8, 4)); pd.Series(best_ga_params).plot(kind='bar');
plt.ylabel('Selected parameter value'); show_plot('GA-selected Random Forest hyperparameters')

# 16. Validation vs test macro F1
combined_macro = pd.DataFrame({
    'Validation Macro F1': all_validation['Macro F1'],
    'Test Macro F1': test_comparison['Macro F1']
}).dropna()
plt.figure(figsize=(9, 4)); combined_macro.plot(kind='bar', figsize=(10,4));
plt.ylabel('Macro F1'); plt.xticks(rotation=35, ha='right'); show_plot('Validation vs test macro F1')


# ## Phase 9: Feature importance and explainability
# 
# Feature importance is reported for the GA-optimized Random Forest model using BPSO-selected features.

# In[17]:


# Random Forest feature importance
bpso_mask = np.isin(search_feature_names, bpso_features)
bpso_col_indices = get_encoded_column_mask(bpso_mask)

try:
    cat_encoded_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
except AttributeError:
    cat_encoded_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names(cat_cols)

encoded_feature_names = np.array(num_cols + list(cat_encoded_names))[bpso_col_indices]

importance = pd.Series(ga_rf_model.feature_importances_, index=encoded_feature_names).sort_values(ascending=False).head(20)
plt.figure(figsize=(8, max(4, len(importance)*0.3)))
sns.barplot(x=importance.values, y=importance.index)
plt.xlabel('Gini Importance'); plt.ylabel('Feature'); plt.title('GA + BPSO Random Forest Top 20 Feature Importances')
plt.tight_layout(); plt.show()
display(importance.to_frame('Importance'))


# In[18]:


try:
    import shap
    shap_n = min(500, len(X_test_enc))
    rng = np.random.default_rng(RANDOM_STATE)
    shap_idx = rng.choice(len(X_test_enc), size=shap_n, replace=False)
    X_shap = get_feature_subset_matrix(X_test_enc, bpso_features)[shap_idx].toarray() if hasattr(X_test_enc, "toarray") else get_feature_subset_matrix(X_test_enc, bpso_features)[shap_idx]
    explainer = shap.TreeExplainer(ga_rf_model)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):
        shap_to_plot = shap_values[1]
    else:
        shap_to_plot = shap_values
    shap.summary_plot(shap_to_plot, X_shap, feature_names=encoded_feature_names, show=True)
except Exception as e:
    print('SHAP analysis could not be completed in this runtime:', repr(e))


# ## Phase 10: Save models, metrics and experiment outputs

# In[19]:


import joblib
from pathlib import Path

OUTPUT_DIR = Path('victorian_crash_experiment_outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

joblib.dump(rf_model, OUTPUT_DIR / 'random_forest_baseline.joblib')
joblib.dump(xgb_model, OUTPUT_DIR / 'xgboost.joblib')
joblib.dump(lgb_model, OUTPUT_DIR / 'lightgbm.joblib')
cb_model.save_model(str(OUTPUT_DIR / 'catboost.cbm'))
joblib.dump(ga_rf_model, OUTPUT_DIR / 'ga_bpso_random_forest.joblib')
joblib.dump(preprocessor, OUTPUT_DIR / 'onehot_preprocessor.joblib')

baseline_table.to_csv(OUTPUT_DIR / 'baseline_validation_metrics.csv')
all_validation.to_csv(OUTPUT_DIR / 'all_validation_metrics.csv')
test_comparison.to_csv(OUTPUT_DIR / 'test_metrics_comparison.csv')
pd.DataFrame({'ACO_Selected_Feature': pd.Series(aco_features),
              'BPSO_Selected_Feature': pd.Series(bpso_features)}).to_csv(OUTPUT_DIR / 'selected_features.csv', index=False)
with open(OUTPUT_DIR / 'ga_rf_best_parameters.json', 'w') as f:
    json.dump(best_ga_params, f, indent=2)

print('Saved experiment outputs to:', OUTPUT_DIR.resolve())


# In[ ]:




