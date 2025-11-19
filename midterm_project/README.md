# Employee Attrition & Performance

## 1. Problem Statement

This project aims to predict employee attrition—whether an employee will leave the company. We will build a binary classification model that uses employee attributes (such as job role, satisfaction, income, and work-life balance) to identify employees at a high risk of attrition.

### 🎯 What We Are Predicting
- **`0`** - Employee will **stay**
- **`1`** - Employee will **leave** (*positive class*)

### 👥 Who Benefits
- **HR Departments & Management**: Proactively identify at-risk employees
- **Company Leadership**: Reduce costs associated with recruitment and training while retaining institutional knowledge

### 💡 How the Model Will Be Used
The model will be deployed as a proactive HR tool generating monthly "attrition risk scores" to:

- Conduct **targeted retention interviews**
- Develop **personalized retention plans** (compensation, role adjustments, training)
- Understand **key drivers** of attrition (job satisfaction, overtime, etc.)

---

## 2. Evaluation Metric

### 🎪 Primary Metric: **Area Under the Precision-Recall Curve (AUC-PR)**

#### 📊 Why This Metric?

**Dataset Characteristics:**
- **Class 0 (Stay)**: ~84% of employees
- **Class 1 (Leave)**: ~16% of employees

**Key Considerations:**
- 🚫 **Accuracy is misleading** - A naive "always predict stay" model would achieve 84% accuracy but catch zero attrition cases
- ✅ **Precision & Recall are critical**:
  - **Precision**: Avoid wasting HR resources on false alarms
  - **Recall**: Catch as many at-risk employees as possible
- 📈 **PR Curve** directly visualizes the precision-recall trade-off
- 🎯 **AUC-PR** provides a single performance measure optimized for imbalanced datasets

#### 📋 Secondary Metrics
- **Confusion Matrix** - Visualize TP, FP, TN, FN
- **Precision & Recall** at optimal threshold
- **F1-Score** - Balanced measure of precision and recall

---

## 3. Business Impact & Why It Matters

### 💰 The Financial Impact
Employee replacement costs range from **½× to 2× annual salary**, translating to **millions in losses** for medium-to-large organizations through:

- Lost productivity and operational disruption
- Recruitment agency fees and hiring costs
- Training and onboarding expenses
- Knowledge transfer inefficiencies

### 🏢 Beyond Financial Costs
- **Team disruption** and morale deterioration
- **Institutional knowledge loss**
- **Customer relationship damage** from key personnel departures

### 🔄 Shifting from Reactive to Proactive
This model enables organizations to move from:

> ❌ *"Why did this person leave?"* (reactive)
>
> ✅ *"How can we prevent this person from leaving?"* (proactive)

**Outcome**: Data-driven retention strategies, improved employee satisfaction, cost reduction, and stable, productive work environments.