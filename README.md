# Comparative Analysis of Machine Learning and Bio-Inspired Optimization Algorithms for Road Accident Severity Prediction

## 1. Project Overview

Road accidents are a major public safety concern, and accurately predicting accident severity can help identify high-risk situations, improve road safety planning, and support data-driven decision-making.

This project presents a comparative analysis of traditional Machine Learning algorithms and bio-inspired optimization techniques for predicting road accident severity using the **Victorian Road Crash Dataset**.

The primary objective is to develop and evaluate multiple classification models to distinguish between fatal and non-fatal road accidents. In addition to conventional machine learning approaches, the project investigates whether bio-inspired optimization algorithms can improve predictive performance through feature selection and hyperparameter optimization.

The study combines:

- Traditional Machine Learning algorithms.
- Ant Colony Optimization (ACO) for feature selection.
- Binary Particle Swarm Optimization (BPSO) for feature selection.
- Genetic Algorithm (GA) for hyperparameter optimization.
- Model explainability using feature importance and SHAP.

## 2. Research Objectives

The major objectives of this project are:

1. Analyze and understand the road accident dataset through exploratory data analysis.
2. Identify relevant accident-related features that influence fatal accident outcomes.
3. Develop baseline classification models using different machine learning algorithms.
4. Address class imbalance between fatal and non-fatal accident records.
5. Apply bio-inspired optimization algorithms to identify relevant input features.
6. Optimize Random Forest hyperparameters using a Genetic Algorithm.
7. Compare conventional machine learning models against optimization-based models.
8. Evaluate model performance using multiple classification metrics.
9. Interpret model predictions using feature importance and SHAP analysis.

## 3. Dataset Description

**Dataset:** Victorian Road Crash Dataset (CSV format)

The dataset contains road accident records with information about accident circumstances, environmental conditions, road characteristics, and accident outcomes.

The dataset is loaded directly from a CSV file and inspected before preprocessing.

### Dataset inspection includes:

- Total number of records and columns.
- Feature names and data types.
- Missing value analysis.
- Duplicate record identification.
- Target variable identification.
- Target class distribution.
- Potential target leakage identification.

### Target Variable

The original `SEVERITY` column is transformed into a binary classification target.

| Class | Description |
|---|---|
| 0 | Non-fatal accident |
| 1 | Fatal accident |

The binary classification approach combines non-injury, other injury, and serious injury outcomes into the non-fatal class.

This approach addresses the extremely small sample count in certain original severity categories and focuses the research on fatal accident prediction.

## 4. Project Methodology

The project is divided into ten major phases.

### Phase 1: Dataset Loading and Inspection

The original CSV dataset is loaded using Pandas.

The following operations are performed:

- Dataset shape and structure analysis.
- Column data type inspection.
- Missing value identification.
- Duplicate record detection.
- Severity column identification.
- Potential target leakage detection.

### Phase 2: Data Cleaning and Preprocessing

The dataset undergoes preprocessing to make it suitable for machine learning.

**Target transformation:**
- Fatal accidents are assigned class 1.
- All non-fatal accidents are assigned class 0.

**Target leakage prevention:**

Identifiers and post-accident outcome-related columns are excluded, including:

- `ACCIDENT_NO`
- `_id`
- `SEVERITY`
- `FATALITY`
- `SERIOUSINJURY`
- `OTHERINJURY`
- `NONINJURED`
- `INJ_OR_FATAL`
- `TOTAL_PERSONS`

Person and vehicle outcome-related columns are also excluded from the feature set.

**Datetime feature engineering:**

- Accident date is transformed into month and day.
- Accident time is transformed into hour.

**Missing value handling:**

- Numerical features: Median imputation.
- Categorical features: Unknown value replacement and categorical encoding.

**Categorical encoding:**

One-Hot Encoding is used to transform categorical features into numerical representations.

**Dataset splitting:**

The dataset is divided using stratified sampling.

| Dataset | Percentage |
|---|---:|
| Training | 64% |
| Validation | 16% |
| Testing | 20% |

A fixed random state is used to ensure reproducibility.

### Phase 3: Baseline Machine Learning Models

Four conventional machine learning algorithms are implemented and evaluated.

| Algorithm | Description |
|---|---|
| Random Forest | Ensemble learning using multiple decision trees |
| XGBoost | Gradient boosting with regularization |
| LightGBM | Efficient gradient boosting framework |
| CatBoost | Gradient boosting optimized for categorical data |

Class imbalance is addressed using class weighting or positive-class weighting, depending on the algorithm.

These models establish the baseline performance against which the optimized models are compared.

### Phase 4: Bio-Inspired Feature Selection

Two bio-inspired optimization algorithms are implemented to identify relevant input features.

The selected feature subsets are evaluated using a Random Forest classifier.

#### 4.1 Ant Colony Optimization (ACO)

ACO is inspired by the foraging behavior of ants.

In this implementation:

1. Each ant generates a binary feature-selection mask.
2. Features are selected based on pheromone values and heuristic information.
3. Selected features are evaluated using Random Forest.
4. The best-performing feature combination is identified.
5. Pheromone values are updated after each iteration.
6. The process continues until the maximum number of iterations is reached.

The fitness function considers validation Macro F1 score with a penalty for selecting too many features.

#### 4.2 Binary Particle Swarm Optimization (BPSO)

BPSO is inspired by the collective movement of bird flocks and fish schools.

In this implementation:

1. Each particle represents a possible feature-selection combination.
2. Particles maintain binary positions and velocity values.
3. Each position is evaluated using Random Forest.
4. Personal best and global best positions are updated.
5. Sigmoid-based probability determines the next binary position.
6. The process continues until the maximum number of iterations is reached.

The best-performing particle represents the selected feature subset.

### Phase 5: Genetic Algorithm for Hyperparameter Optimization

A Genetic Algorithm is implemented to optimize Random Forest hyperparameters using the BPSO-selected feature subset.

The optimization process consists of:

1. Population initialization.
2. Fitness evaluation.
3. Selection of high-performing individuals.
4. Crossover between parent solutions.
5. Mutation of selected parameters.
6. Elitism to preserve the best solutions.
7. Repeated evolution across generations.

The following hyperparameters are optimized:

| Hyperparameter | Search Range |
|---|---|
| `n_estimators` | 100–400 |
| `max_depth` | 5–24 |
| `min_samples_split` | 2–14 |
| `min_samples_leaf` | 1–9 |
| `max_features` | 0.3–1.0 |

The fitness function is based on validation Macro F1 score.

### Phase 6: Optimized Model Training

The optimization-based feature subsets are used to train and evaluate three models:

1. ACO + Random Forest
2. BPSO + Random Forest
3. GA + BPSO + Random Forest

The Genetic Algorithm optimized Random Forest uses the BPSO-selected feature subset and optimized hyperparameters.

The performance of these models is compared against the four baseline machine learning models.

### Phase 7: Final Test Evaluation

All trained models are evaluated using the independent held-out test dataset.

The test dataset is not used during model training or optimization.

The final comparison includes all seven model configurations.

The best model is selected using validation Macro F1 score rather than test-set performance.

### Phase 8: Performance Evaluation and Visualization

Multiple evaluation metrics are used to assess model performance.

| Metric | Purpose |
|---|---|
| Accuracy | Overall prediction correctness |
| Balanced Accuracy | Performance across imbalanced classes |
| Macro F1 | Equal importance to both classes |
| Weighted F1 | F1 score weighted by class frequency |
| Fatal Precision | Correctness of fatal predictions |
| Fatal Recall | Ability to identify actual fatal accidents |
| Fatal F1 | Balance between fatal precision and recall |
| ROC AUC | Discrimination between the two classes |
| PR AUC | Precision-recall performance, particularly useful for imbalanced data |

The following visualizations are generated:

- Target class distribution.
- Missing value distribution.
- Numerical feature distributions.
- Categorical feature distributions.
- ACO convergence graph.
- BPSO convergence graph.
- Genetic Algorithm convergence graph.
- Model performance comparison.
- Confusion matrices.
- ROC curves.
- Precision-Recall curves.
- Feature selection comparison.
- Validation versus test performance comparison.

### Phase 9: Feature Importance and Explainability

Model interpretability is investigated using two approaches.

**Random Forest Feature Importance**

The feature importance values of the GA-optimized Random Forest model are extracted to identify the most influential encoded features.

The top 20 features are visualized.

**SHAP (SHapley Additive exPlanations)**

SHAP is used to understand the contribution of individual features to model predictions.

The analysis uses a sample of test observations and a TreeExplainer for the optimized Random Forest model.

The SHAP summary plot provides insights into:

- Feature contribution to predictions.
- Relative feature importance.
- Positive and negative effects on model output.
- Differences in feature influence across observations.

### Phase 10: Saving Experiment Outputs

All trained models and experiment results are saved for future analysis and reproducibility.

The output directory is:

`victorian_crash_experiment_outputs/`

The saved artifacts include:

| File | Description |
|---|---|
| `random_forest_baseline.joblib` | Baseline Random Forest model |
| `xgboost.joblib` | XGBoost model |
| `lightgbm.joblib` | LightGBM model |
| `catboost.cbm` | CatBoost model |
| `ga_bpso_random_forest.joblib` | Optimized Random Forest model |
| `onehot_preprocessor.joblib` | Fitted preprocessing pipeline |
| `baseline_validation_metrics.csv` | Baseline validation results |
| `all_validation_metrics.csv` | Complete validation comparison |
| `test_metrics_comparison.csv` | Final test performance comparison |
| `selected_features.csv` | ACO and BPSO selected features |
| `ga_rf_best_parameters.json` | Best Genetic Algorithm parameters |

## 5. Model Comparison

The project evaluates seven model configurations.

| Category | Model |
|---|---|
| Baseline | Random Forest |
| Baseline | XGBoost |
| Baseline | LightGBM |
| Baseline | CatBoost |
| Bio-inspired | ACO + Random Forest |
| Bio-inspired | BPSO + Random Forest |
| Hybrid optimization | GA + BPSO + Random Forest |

This comparison helps investigate the effectiveness of bio-inspired feature selection and hyperparameter optimization compared with conventional machine learning approaches.

Actual performance values are generated during execution and stored in the experiment output files.

## 6. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Dataset manipulation |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning and preprocessing |
| XGBoost | Gradient boosting |
| LightGBM | Gradient boosting |
| CatBoost | Categorical boosting |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| SHAP | Model explainability |
| Joblib | Model serialization |
| Jupyter Notebook | Experiment execution |

## 7. Installation and Setup

### Prerequisites

- Python 3.9 or later.
- Jupyter Notebook or JupyterLab.
- CSV dataset.
- Recommended: 8 GB or more RAM for large dataset experiments.

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install xgboost lightgbm catboost
pip install shap imbalanced-learn joblib jupyter
```

### Step 4: Configure Dataset Path

Update the CSV path in the notebook:

```python
csv_path = "path/to/your/dataset.csv"
```

### Step 5: Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the project notebook and execute the cells sequentially.

## 8. Project Workflow

```text
              Road Accident Dataset
                       |
                       v
             Dataset Inspection
                       |
                       v
             Data Preprocessing
                       |
                       v
             Feature Engineering
                       |
                       v
               Stratified Split
                       |
              +--------+--------+
              |                 |
              v                 v
       Baseline Models    Bio-Inspired Search
              |                 |
              |          +------+------+
              |          |             |
              |          v             v
              |         ACO           BPSO
              |          |             |
              |          v             v
              |     RF Evaluation  RF Evaluation
              |          |             |
              |          v             v
              |    Selected Features   |
              |                        v
              |                 Genetic Algorithm
              |                        |
              |                        v
              |               Optimized Random Forest
              |                        |
              +-----------+------------+
                          |
                          v
                 Model Comparison
                          |
                          v
                 Final Test Evaluation
                          |
                          v
                Explainability (SHAP)
                          |
                          v
                 Save Experiment Results
```

## 9. Research Significance

This project investigates the application of bio-inspired optimization techniques to road accident severity prediction.

Rather than relying exclusively on conventional machine learning algorithms, it explores the combination of:

- Nature-inspired feature selection.
- Evolutionary hyperparameter optimization.
- Ensemble-based classification.
- Explainable machine learning.

The comparative experimental framework provides a way to evaluate whether optimization techniques can improve predictive performance, reduce feature dimensionality, and support more interpretable accident severity prediction.

The project is intended for academic research and analytical purposes. Predictions should not be treated as definitive assessments of accident risk or used as a substitute for validated road safety systems.

## 10. Future Enhancements

Potential future improvements include:

- Incorporating additional road, traffic, and weather datasets.
- Evaluating temporal and geographical accident patterns.
- Experimenting with SMOTE and other imbalance-handling techniques within training folds.
- Implementing additional optimization algorithms such as Artificial Bee Colony and Grey Wolf Optimizer.
- Performing systematic hyperparameter tuning using cross-validation.
- Introducing probability calibration and decision-threshold optimization.
- Applying SHAP analysis across multiple models.
- Developing an interactive dashboard for accident severity prediction.
- Conducting statistical significance testing across repeated experiments.
- Evaluating generalization using time-based and geographical holdout datasets.

## 11. Important Research Considerations

For reliable experimental conclusions, the following considerations should be addressed:

1. Feature selection and hyperparameter optimization must use training and validation data only.
2. Imputation and preprocessing should ideally be fitted exclusively on training data.
3. The held-out test dataset must remain untouched until final evaluation.
4. Fatal accident recall and Precision-Recall AUC should receive particular attention because fatal cases represent a minority class.
5. Feature importance indicates predictive association, not necessarily causation.
6. The original severity mapping and excluded columns should be verified against the dataset documentation.
7. Repeated experiments and cross-validation can help establish the stability of reported performance.

## 12. Conclusion

The project establishes a comparative machine learning framework for binary road accident severity prediction.

By combining Random Forest, XGBoost, LightGBM, and CatBoost with Ant Colony Optimization, Binary Particle Swarm Optimization, and Genetic Algorithms, the study investigates different approaches to feature selection and model optimization.

The final evaluation compares seven model configurations using multiple performance metrics, supported by visual analysis and SHAP-based explainability.

The experimental results can help determine whether bio-inspired optimization provides measurable benefits over conventional machine learning approaches for predicting fatal road accident outcomes.

---

**Project Title:** Comparative Analysis of Machine Learning and Bio-Inspired Optimization Algorithms for Road Accident Severity Prediction

**Domain:** Machine Learning | Road Safety Analytics | Bio-Inspired Optimization | Explainable AI

**Implementation:** Python | Scikit-learn | XGBoost | LightGBM | CatBoost | ACO | BPSO | Genetic Algorithm | SHAP
