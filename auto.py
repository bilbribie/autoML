import autosklearn.classification
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import sklearn.datasets
import sklearn.metrics

# run model
def model(data , target_column):
    
    X = data.drop(target_column, axis=1)  # Features: all columns except the target
    y = data[target_column]  # Target: the column named by 'target_column'
    
    # Split the dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=1)

    # Create an AutoSklearn classifier
    classifier = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=30)

    # Fit the classifier
    classifier.fit(X_train, y_train)
    
    # Predictions
    y_hat = classifier.predict(X_test)
    pred_proba = classifier.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

    # Metrics
    accuracy = accuracy_score(y_test, y_hat)
    report = classification_report(y_test, y_hat)
    auc = roc_auc_score(y_test, pred_proba) if len(set(y)) == 2 else "N/A"  # AUC only for binary targets
    
    return accuracy, report, auc, classifier

# output
def output(accuracy, classification_report, auc, classifier):
    
    return 

if __name__ == "__main__":
    print("start running")
    data = pd.read_csv('Dataset_normalized.csv')  
    
    # select categories
    target_column = "Generic policy", "Reporting mechanism", "Scope of practice", "User guideline"
    #categories = ["Generic policy", "Reporting mechanism", "Scope of practice", "User guideline"]
    
    # select features
    #features = "num_commits", "project_age_days", "num_issues", "num_pull"
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
    
    #data_selected = data[list(features) + [target_column]] #select data

    #run model
    accuracy, classification_report, auc, classifier = model(data, target_column)
    
    #output
    output(accuracy, classification_report, auc, classifier)
    
    # print(f"#########{target_column}:{features}#############")
    print(f"accuracy: {accuracy}")
    print(f"Average auc Score: {auc}")
    print(f"report: {classification_report}")
    print(f"Average auc Score: {auc}")
    print(f"best model:")
    print(classifier.leaderboard())
    
