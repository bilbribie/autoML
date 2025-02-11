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
import re


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
    report = classification_report(y_test, y_hat, output_dict=True)
    macro_avg_f1 = report['macro avg']['f1-score']
    auc = roc_auc_score(y_test, pred_proba) if len(set(y)) == 2 else "N/A"  # AUC only for binary targets
    
    return accuracy, report, auc, classifier, X_test, macro_avg_f1

# find model 1st rank
def get_model(classifier):
    model_descriptions = classifier.show_models()
    models_dict = model_descriptions 
    
    # Find the model with rank 1
    model_info = next((info for info in models_dict.values() if info['rank'] == 1), None)
    
    if model_info:
        # Extract `sklearn_regressor` as a string representation
        sklearn_regressor = str(model_info['sklearn_regressor'])
        
        # Extract only the model name using regex
        model_name_match = re.match(r'(\w+)\(', sklearn_regressor)
        model_name = model_name_match.group(1) + "()" if model_name_match else None
        
        return model_name, sklearn_regressor
    
    return None, None

# SHAP value
def shap_values(model_info, X_test, target_column):
    
    explainer = shap.Explainer(model_info)
    shap_values = explainer(X_test)
    
    # Plotting SHAP values and save in folder
    plt.figure()
    shap.summary_plot(shap_values, X_test)
    plt.savefig(f'pics/{target_column}_shap.png')
    plt.close()
    return 
   
if __name__ == "__main__":
    print("start running")
    data = pd.read_csv('Dataset_Normalized copy.csv')  
    categories = ["Generic policy", "Reporting mechanism", "Scope of practice", "User guideline"]
    
    results = []
    
    for target_column in categories:
        print(f"Processing for {target_column}")
        
        # 1. model train
        data_selected = data.drop([col for col in categories if col != target_column], axis=1)  # Drop other target cols
        accuracy, report, auc, classifier, X_test, macro_avg_f1 = model(data_selected, target_column)
        model_name, sklearn_regressor = get_model(classifier) # get model rank 1
        print(f"The best model for {target_column} is {model_name}")
        
        # 2. SHAP
        shap = shap_values(model_name, X_test, target_column)

        # Print results
        results.append([target_column, accuracy, auc, macro_avg_f1, sklearn_regressor])
        print(f"Finished processing {target_column}. Results: Accuracy={accuracy}, AUC={auc}")
    
    # Output the results
    results_df = pd.DataFrame(results, columns=['Target Column', 'Accuracy', 'AUC', 'Macro Avg F1', 'Best Model'])
    results_df.to_csv('model_results.csv', index=False)
    print("All processing complete. Results saved to model_results.csv.")
    
