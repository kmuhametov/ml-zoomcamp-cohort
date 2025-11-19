# Employee Attrition Prediction

## Problem Statement

### Business Challenge
Employee attrition is a major and costly problem for businesses. This project aims to predict employee attrition—whether an employee will leave the company—using a binary classification model that analyzes employee attributes (job role, satisfaction, income, work-life balance, etc.) to identify high-risk employees.

### 🎯 Prediction Target
- **`0`** - Employee will **stay**
- **`1`** - Employee will **leave** (*positive class*)

### 👥 Stakeholders & Value Proposition
- **HR Departments & Management**: Proactively identify at-risk employees for targeted interventions
- **Company Leadership**: Reduce costs associated with recruitment and training while retaining institutional knowledge

### 💡 Solution Implementation
The model will be deployed as a proactive HR tool generating monthly "attrition risk scores" to enable:

- **Targeted retention interviews** with high-risk employees
- **Personalized retention plans** (compensation adjustments, role changes, training)
- **Root cause analysis** of key attrition drivers (job satisfaction, overtime, etc.)
- **Data-driven decision making** for HR strategy

### 📊 Evaluation Framework

#### Primary Metric: **Area Under the Precision-Recall Curve (AUC-PR)**

**Why AUC-PR for This Problem?**
- **Dataset Characteristics**: 
  - Class 0 (Stay): ~84% of employees
  - Class 1 (Leave): ~16% of employees (imbalanced classification)
- **Accuracy is misleading**: A naive "always predict stay" model would achieve 84% accuracy but catch zero attrition cases
- **Precision & Recall are critical**:
  - **Precision**: Avoid wasting HR resources on false alarms
  - **Recall**: Catch as many at-risk employees as possible
- **PR Curve Advantage**: Directly visualizes the precision-recall trade-off optimized for imbalanced datasets

#### Secondary Metrics
- **Confusion Matrix** - Visualize True Positives, False Positives, True Negatives, False Negatives
- **Precision & Recall** at optimal classification threshold
- **F1-Score** - Balanced harmonic mean of precision and recall

### 💰 Business Impact & Strategic Importance

#### Financial Impact
Employee replacement costs range from **½× to 2× annual salary**, translating to **millions in losses** for organizations through:

- **Direct Costs**: Recruitment agency fees, hiring bonuses, background checks
- **Productivity Loss**: Operational disruption, training time, knowledge transfer
- **Indirect Costs**: Team morale impact, customer relationship damage, institutional knowledge loss

#### Organizational Impact
- **Team disruption** and morale deterioration from frequent turnover
- **Institutional knowledge loss** affecting continuity and quality
- **Customer relationship damage** when key personnel depart
- **Recruitment burden** on remaining staff and managers

#### Strategic Shift
This model enables organizations to transition from:

> ❌ *"Why did this person leave?"* (reactive post-mortem)
>
> ✅ *"How can we prevent this person from leaving?"* (proactive retention)

**Final Outcome**: Data-driven retention strategies, improved employee satisfaction, significant cost reduction, and stable, productive work environments.

## Dataset Description

### 📁 Dataset Overview

**Source**: [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

| Aspect | Details |
|--------|---------|
| **Total Records** | 1,470 employees |
| **Time Period** | Current workforce snapshot |
| **Attrition Rate** | 16.1% (237 left, 1,233 stayed) |
| **Number of Features** | 35 attributes per employee |
| **Data Type** | Structured tabular data |

### 📊 Target Variable

| Variable | Description | Distribution |
|----------|-------------|--------------|
| **Attrition** | Whether employee left the company | **Yes**: 16.1% (237)<br>**No**: 83.9% (1,233) |

### 🗂️ Feature Categories

#### 1. **Demographic Information**
| Feature | Type | Description |
|---------|------|-------------|
| Age | Numerical | Employee's age in years |
| Gender | Categorical | Male/Female |
| MaritalStatus | Categorical | Single, Married, Divorced |
| Education | Numerical | 1 'Below College' - 5 'Doctor' |

#### 2. **Job & Organizational Context**
| Feature | Type | Description |
|---------|------|-------------|
| Department | Categorical | HR, R&D, Sales |
| JobRole | Categorical | 9 unique roles |
| JobLevel | Numerical | 1-5 (1: lowest) |
| YearsAtCompany | Numerical | Total years at company |
| YearsInCurrentRole | Numerical | Years in current position |

#### 3. **Compensation & Benefits**
| Feature | Type | Description |
|---------|------|-------------|
| MonthlyIncome | Numerical | Current monthly salary |
| DailyRate | Numerical | Standard daily rate |
| HourlyRate | Numerical | Standard hourly rate |
| StockOptionLevel | Numerical | 0-3 (0: lowest) |
| PercentSalaryHike | Numerical | Last salary increase percentage |

#### 4. **Satisfaction & Work Environment**
| Feature | Type | Description |
|---------|------|-------------|
| JobSatisfaction | Numerical | 1-4 (1: low, 4: high) |
| EnvironmentSatisfaction | Numerical | 1-4 (1: low, 4: high) |
| RelationshipSatisfaction | Numerical | 1-4 (1: low, 4: high) |
| WorkLifeBalance | Numerical | 1-4 (1: low, 4: high) |

#### 5. **Work Patterns & Behavior**
| Feature | Type | Description |
|---------|------|-------------|
| OverTime | Categorical | Yes/No |
| BusinessTravel | Categorical | Non-Travel, Travel_Rarely, Travel_Frequently |
| TrainingTimesLastYear | Numerical | Number of training sessions |
| NumCompaniesWorked | Numerical | Total companies worked for |

### 📈 Key Data Characteristics

#### Class Distribution
```python
# Attrition distribution
Attrition_No     1233 (83.9%)
Attrition_Yes     237 (16.1%)

# Employee Attrition Prediction

## Problem Statement

### Business Challenge
Employee attrition is a major and costly problem for businesses. This project aims to predict employee attrition—whether an employee will leave the company—using a binary classification model that analyzes employee attributes to identify high-risk employees.

### 🎯 Prediction Target
- **`0`** - Employee will **stay**
- **`1`** - Employee will **leave** (*positive class*)

### 👥 Stakeholders & Value Proposition
- **HR Departments & Management**: Proactively identify at-risk employees for targeted interventions
- **Company Leadership**: Reduce costs associated with recruitment and training while retaining institutional knowledge

### 💡 Solution Implementation
The model will be deployed as a proactive HR tool generating monthly "attrition risk scores" to enable:
- **Targeted retention interviews** with high-risk employees
- **Personalized retention plans** (compensation adjustments, role changes, training)
- **Root cause analysis** of key attrition drivers
- **Data-driven decision making** for HR strategy

### 📊 Evaluation Framework

#### Primary Metric: **Area Under the Precision-Recall Curve (AUC-PR)**
**Why AUC-PR for This Problem?**
- **Dataset Characteristics**: 
  - Class 0 (Stay): ~84% of employees
  - Class 1 (Leave): ~16% of employees (imbalanced classification)
- **Accuracy is misleading**: A naive "always predict stay" model would achieve 84% accuracy but catch zero attrition cases
- **Precision & Recall are critical**:
  - **Precision**: Avoid wasting HR resources on false alarms
  - **Recall**: Catch as many at-risk employees as possible

#### Secondary Metrics
- **Confusion Matrix** - Visualize TP, FP, TN, FN
- **Precision & Recall** at optimal classification threshold
- **F1-Score** - Balanced harmonic mean of precision and recall

### 💰 Business Impact & Strategic Importance

#### Financial Impact
Employee replacement costs range from **½× to 2× annual salary**, translating to **millions in losses** for organizations through:
- **Direct Costs**: Recruitment fees, hiring bonuses, background checks
- **Productivity Loss**: Operational disruption, training time, knowledge transfer
- **Indirect Costs**: Team morale impact, institutional knowledge loss

#### Strategic Shift
This model enables organizations to transition from:
> ❌ *"Why did this person leave?"* (reactive)
>
> ✅ *"How can we prevent this person from leaving?"* (proactive)

---

## Dataset Description

### 📁 Dataset Overview

| Aspect | Details |
|--------|---------|
| **Total Records** | 1,470 employees |
| **Time Period** | Current workforce snapshot |
| **Attrition Rate** | 16.1% (237 left, 1,233 stayed) |
| **Number of Features** | 35 attributes per employee |
| **Data Type** | Structured tabular data |

### 📊 Target Variable

| Variable | Description | Distribution |
|----------|-------------|--------------|
| **Attrition** | Whether employee left the company | **Yes**: 16.1% (237)<br>**No**: 83.9% (1,233) |

### 🗂️ Feature Categories

| Category | Features | Description |
|----------|----------|-------------|
| **Demographic** | Age, Gender, Marital Status | Employee background |
| **Job Context** | Department, Job Role, Job Level | Organizational position |
| **Compensation** | Monthly Income, Stock Options | Financial incentives |
| **Satisfaction** | Job, Environment, Relationship | Employee sentiment |
| **Work Patterns** | Overtime, Business Travel | Work behaviors |

### 🎯 Data Suitability

| Aspect | Assessment |
|--------|------------|
| **Feature Richness** | ✅ Comprehensive employee attributes |
| **Data Quality** | ✅ No missing values, clean structure |
| **Class Balance** | ⚠️ Imbalanced (16% positive class) |
| **Predictive Power** | ✅ Strong correlations with target |

---

## EDA Summary

### 📊 Data Overview & Initial Analysis

#### Dataset Structure
- **1,470 employees** with **35 features** capturing comprehensive workforce attributes
- **Clean dataset** with no missing values across all features
- **Target variable**: `Attrition` (16.1% positive class - imbalanced classification)

#### Key Transformations Applied
- Converted target variable from categorical ('Yes'/'No') to binary (1/0)
- Identified and separated **8 categorical** and **27 numerical** features
- Handled potential missing values with appropriate imputation strategies

### 🔍 Feature Analysis & Selection

#### Numerical Features Correlation
**Top Correlated Features with Attrition:**

| Feature | Correlation | Business Insight |
|---------|-------------|------------------|
| **TotalWorkingYears** | 0.171 | Career experience impact |
| **JobLevel** | 0.169 | Organizational hierarchy |
| **YearsInCurrentRole** | 0.161 | Role stagnation factor |
| **MonthlyIncome** | 0.160 | Compensation influence |
| **Age** | 0.159 | Demographic factor |

**Features Removed** (correlation < 0.10):
```python
['DistanceFromHome', 'WorkLifeBalance', 'TrainingTimesLastYear', 'DailyRate', 
 'RelationshipSatisfaction', 'NumCompaniesWorked', 'YearsSinceLastPromotion', 
 'Education', 'MonthlyRate', 'PercentSalaryHike', 'EmployeeNumber', 
 'HourlyRate', 'PerformanceRating', 'EmployeeCount', 'StandardHours']

 # Modeling Approach & Metrics

## Model Selection Strategy

Three different machine learning models were implemented and compared to predict employee attrition:

1. **Logistic Regression** - A linear model for binary classification
2. **Random Forest** - An ensemble method using multiple decision trees
3. **Gradient Boosting** - A sequential ensemble technique that builds trees to correct previous errors

## Model Training & Hyperparameter Tuning

The models were trained using **5-fold cross-validation** with **GridSearchCV** to find optimal hyperparameters:

### Parameter Grids

**Logistic Regression:**
- `C`: [0.01, 0.1, 1, 10, 100]
- `solver`: ['lbfgs', 'liblinear']
- `class_weight`: [None, 'balanced']

**Random Forest:**
- `n_estimators`: [50, 100, 200]
- `max_depth`: [10, 20, None]
- `min_samples_split`: [2, 5, 10]
- `class_weight`: [None, 'balanced']

**Gradient Boosting:**
- `n_estimators`: [50, 100, 200]
- `max_depth`: [3, 6, 9]
- `learning_rate`: [0.01, 0.1, 0.3]
- `subsample`: [0.8, 1.0]

## Evaluation Metrics

Models were evaluated using multiple metrics:
- **ROC-AUC**: Area Under the ROC Curve
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of precision and recall

## Results Summary

### Cross-Validation Performance
| Model | Best CV ROC-AUC | Best Parameters |
|-------|-----------------|-----------------|
| Logistic Regression | 0.7955 | {'C': 0.1, 'class_weight': None, 'solver': 'lbfgs'} |
| Random Forest | 0.8029 | {'class_weight': None, 'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 200} |
| Gradient Boosting | 0.8010 | {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 200, 'subsample': 0.8} |

### Validation Set Performance
| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|-----------|
| Logistic Regression | 0.8049 | 0.8462 | 0.2500 | 0.3860 |
| Random Forest | 0.7958 | 0.6429 | 0.2045 | 0.3103 |
| Gradient Boosting | 0.7626 | 0.6667 | 0.3182 | 0.4308 |

### Test Set Performance
| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|-----------|
| Logistic Regression | 0.8301 | 0.9375 | 0.2586 | 0.4054 |
| Random Forest | 0.7938 | 0.7368 | 0.2414 | 0.3636 |
| Gradient Boosting | 0.7803 | 0.5357 | 0.2586 | 0.3488 |

## Final Model Selection

**🎯 Best Performing Model: Logistic Regression**
- **Test ROC-AUC**: 0.8301
- **Test F1-Score**: 0.4054

Despite having slightly lower cross-validation scores than ensemble methods, Logistic Regression demonstrated the best generalization performance on the test set with the highest ROC-AUC score.

## Feature Importance Analysis

### Top 10 Features - Random Forest:
1. MonthlyIncome (11.76%)
2. Age (8.71%)
3. YearsAtCompany (6.91%)
4. TotalWorkingYears (6.84%)
5. YearsWithCurrManager (5.14%)
6. JobInvolvement (4.95%)
7. OverTime=Yes (4.86%)
8. EnvironmentSatisfaction (4.78%)
9. YearsInCurrentRole (4.71%)
10. OverTime=No (4.34%)

### Top 10 Features - Gradient Boosting:
1. MonthlyIncome (19.24%)
2. Age (11.27%)
3. YearsAtCompany (7.76%)
4. TotalWorkingYears (6.12%)
5. JobInvolvement (5.63%)
6. OverTime=No (4.12%)
7. EnvironmentSatisfaction (4.06%)
8. YearsWithCurrManager (4.01%)
9. StockOptionLevel (3.98%)
10. OverTime=Yes (3.53%)

**Key Insights**: Monthly income and age are consistently the most important predictors across both ensemble methods, followed by employment tenure-related features.