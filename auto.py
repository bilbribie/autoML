import autosklearn.classification
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import sklearn.metrics
import subprocess


# run model
def model(data , target_column):
    
    X = data.drop(target_column, axis=1)  # Features: all columns except the target
    y = data[target_column]  # Target: the column named by 'target_column'
    
    # Split the dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

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
    
    data = {
        
    }
    return 

results = []
    
   
if __name__ == "__main__":
    print("start running")
    data = pd.read_csv('Dataset_eiei.csv')  
    categories = ["Reporting mechanism", "Scope of practice", "User guideline"]
    
    for target_column in categories:
        #model train
        data_selected = data.drop([col for col in categories if col != target_column], axis=1)  # Drop other target cols
        accuracy, report, auc, classifier, X_test = model(data_selected, target_column)

        # SHAP values
        explainer = shap.TreeExplainer(classifier.show_models().estimators_[0][0])  # Using the first estimator in the ensemble
        shap_values = explainer.shap_values(X_test)
        
        # Plot and save SHAP values
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(f'pics/{target_column}_shap.png')
        plt.close()
        
        # Save results
        macro_avg_f1 = report['macro avg']['f1-score']
        best_model_details = classifier.show_models().models_[0]  # Assuming the first model is the best
        
        results.append([target_column, accuracy, auc, macro_avg_f1, str(best_model_details)])

        # Print results
        print(f"Finished processing {target_column}. Results: Accuracy={accuracy}, AUC={auc}")
    
    # print(f"#########{target_column}:{features}#############")
    print(f"accuracy: {accuracy}")
    print(f"Average auc Score: {auc}")
    print(f"report: {classification_report}")
    print(f"best model:")
    print(classifier.leaderboard())
    
