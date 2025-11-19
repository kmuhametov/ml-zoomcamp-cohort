import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Loading and preparing data...")
    # Load data
    df = pd.read_csv('data/data_prepared.csv')
    
    # Prepare features
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical.remove("Attrition")
    
    # Split data
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
    
    # Reset indices and prepare target variables
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    y_train = df_train.Attrition.values
    y_val = df_val.Attrition.values
    y_test = df_test.Attrition.values
    
    del df_train['Attrition']
    del df_val['Attrition']
    del df_test['Attrition']
    
    # Prepare features with DictVectorizer
    dv = DictVectorizer(sparse=False)
    
    train_dict = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    
    val_dict = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dict)
    
    test_dict = df_test[categorical + numerical].to_dict(orient='records')
    X_test = dv.transform(test_dict)
    
    print("Training best model (Logistic Regression) with optimal parameters...")
    # Best parameters from the notebook
    best_params = {'C': 0.1, 'class_weight': None, 'solver': 'lbfgs'}
    
    # Train the best model
    best_model = LogisticRegression(**best_params, max_iter=10000, random_state=1)
    best_model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_val_pred = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred)
    
    # Evaluate on test set
    y_test_pred = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred)
    
    print(f"Validation ROC-AUC: {val_auc:.4f}")
    print(f"Test ROC-AUC: {test_auc:.4f}")
    
    # Save the model and vectorizer
    print("Saving model and vectorizer...")
    model_data = {
        'model': best_model,
        'dv': dv,
        'categorical': categorical,
        'numerical': numerical
    }
    
    joblib.dump(model_data, 'models/best_model.pkl')
    print("Model saved as 'models/best_model.pkl'")
    
    # Print feature importance (coefficients)
    print("\nTop 10 Most Important Features:")
    feature_names = dv.get_feature_names_out()
    coefficients = best_model.coef_[0]
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(coefficients)
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))

if __name__ == "__main__":
    main()