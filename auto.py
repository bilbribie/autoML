import autosklearn.classification
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

# Main function to run auto-sklearn multiple times and calculate averages
def model(data, target_column):
    
    # Load a dataset
    X = data
    y = target_column

    pred = model.predict_proba(X_test)[:, 1] 
    # Split the dataset into training and testing data
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

    # Create an AutoSklearn classifier
    model = autosklearn.classification.AutoSklearnClassifier()
    #time_left_for_this_task=120, per_run_time_limit=30

    # Fit the model
    model.fit(X_train, y_train)
    
    y_hat = model.predict(X_test) # prediction
    
    accuracy = sklearn.metrics.accuracy_score(y_test, y_hat)
    #print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))
    
    #   Evaluate the model
    classification_report(y_test, y_hat)
    #print(classification_report)
    
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    auc = metrics.auc(fpr, tpr)
    #print("AUC:", auc)
    
    return accuracy, classification_report, auc

if __name__ == "__main__":
    data = pd.read_csv('Dataset_normalized.csv')  
    
    # select categories
    target_column = 'Generic policy' 
    #categories = ["Generic policy", "Reporting mechanism", "Scope of practice", "User guideline"]
    
    # select features
    features = "num_commits", "project_age_days", "num_issues", "num_pull"
    # feature_types = {
    # "activeness": ["num_commits", "project_age_days", "num_issues", "num_pull"],
    # "popularity": ["num_stargazers", "num_watchers", "num_forks", "num_subscribers"],
    # "metadata": ["num_contributors", "project_size(kB)"],
    # "security_practice": ["ssf0_Binary-Artifacts", "ssf1_Branch-Protection",
    #                       "ssf3_CII-Best-Practices", "ssf7_Dependency-Update-Tool",
    #                       "ssf8_Fuzzing", "ssf9_License", "ssf10_Maintained", "ssf13_SAST",
    #                       "ssf17_Vulnerabilities"],
    # #    "project_quality": ["sonarQube_BUG", "sonarQube_VULNERABILITY", "sonarQube_CODE_SMELL"],
    # }
    
    data_selected = data[list(features) + [target_column]]

    accuracy, classification_report, auc = model(data_selected , target_column)
    print(f"accuracy: {accuracy}")
    print(f"report: {classification_report}")
    print(f"Average auc Score: {auc}")
    
