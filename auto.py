import autosklearn.classification
import shap
shap.initjs()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from matplotlib.colors import LinearSegmentedColormap
from imblearn.over_sampling import SMOTE

# Define feature types
feature_types = {
    "Project_feature": ["num_commits", "project_age_days", "num_issues", "num_pull", 
                        "num_stargazers", "num_watchers", "num_forks", "num_subscribers", 
                        "num_contributors", "project_size(kB)"],
    "security_practice": ["ssf0_Binary-Artifacts", "ssf1_Branch-Protection",
                          "ssf3_CII-Best-Practices", "ssf7_Dependency-Update-Tool",
                          "ssf8_Fuzzing", "ssf9_License", "ssf10_Maintained", "ssf13_SAST",
                          "ssf17_Vulnerabilities"],
    "project_quality": ['num_sonarQube_BUG_HIGH', 'num_sonarQube_BUG_MEDIUM', 'num_sonarQube_BUG_LOW', 'num_sonarQube_BUG_BLOCKER',
            'num_sonarQube_VULNERABILITY_HIGH', 'num_sonarQube_VULNERABILITY_MEDIUM', 'num_sonarQube_VULNERABILITY_LOW',
            'num_sonarQube_VULNERABILITY_BLOCKER', 'num_sonarQube_CODE_SMELL_HIGH', 'num_sonarQube_CODE_SMELL_MEDIUM',
            'num_sonarQube_CODE_SMELL_LOW', 'num_sonarQube_CODE_SMELL_BLOCKER'],
}

# Feature selection
def select_features_chi2(data, target_column, top_n=7):
    
    print(f"Selecting top {top_n} features for {target_column} using Chi-squared test...")
    
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Apply the Chi-squared test
    chi2_selector = SelectKBest(chi2, k=top_n)
    X_new = chi2_selector.fit_transform(X, y)
    print(X_new)
    
    # Get selected feature names
    selected_features = X.columns[chi2_selector.get_support()].tolist()
    
    print(f"Selected features for {target_column}: {selected_features}")
    
    return selected_features

# run model
def model(data , target_column):
    
    X = data.drop(target_column, axis=1)  # Features: all columns except the target
    y = data[target_column]  # Target: the column named by 'target_column'
    
    # Split the dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Create an AutoSklearn classifier
    classifier = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=30) #time_left_for_this_task=30

    # Fit the classifier
    classifier.fit(X_train, y_train)

    # Predictions
    y_pred = classifier.predict(X_test)
    pred_proba = classifier.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_pred, y_test, output_dict=True)
    report1 = classification_report(y_pred, y_test)
    macro_avg_f1 = report['macro avg']['f1-score']
    auc = roc_auc_score(y_test, pred_proba) if len(set(y)) == 2 else "N/A"  # AUC only for binary targets
    print(f"Accuracy: {accuracy}")
    print(f"Macro avg: {macro_avg_f1}")
    print(f"AUC score: {auc}")
    print("Classification report:")
    print(report1)
    
    return accuracy, report, auc, classifier, macro_avg_f1, X_train, X_test, y_train, y_test

# find model 1st rank

# find model 1st rank
def get_model(models_dict):
    # Find the model with rank 1
    model_info = next((info for info in models_dict.values() if info.get('rank') == 1), None)

    if model_info:
        if 'sklearn_classifier' in model_info:
            sklearn_regressor = model_info['sklearn_classifier']  # Extract the actual model object
        else:
            print("No 'sklearn_classifier' found in model_info keys:", model_info.keys())
            return None, None

        return sklearn_regressor  # Return the model itself

    print("ERROR: No ranked model found")
    return None


# SHAP value
def shap_values(sklearn_regressor,target_column, X_train, X_test, y_train, y_test):
    print(f"Processing SHAP for {target_column} by {sklearn_regressor}")
    model = sklearn_regressor
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # pred_proba = model.predict_proba(X_test)[:, 1] 

    accuracy = accuracy_score(y_test, y_pred)
    # report = classification_report(y_pred, y_test)
    # auc = roc_auc_score(y_test, pred_proba)  # AUC only for binary targets
    print(f"Accuracy: {accuracy}")
    # print(f"AUC score: {auc}")
    # print("Classification report:")
    # print(report)
    
    explainer = shap.KernelExplainer(model.predict, shap.kmeans(X_train, 10))
    shap_values = explainer.shap_values(X_test)
    
    # Plotting SHAP values and save in folder
    cmap = LinearSegmentedColormap.from_list("custom", ["#f3baba", "#006C6C"])
    plt.figure()
    shap.summary_plot(shap_values, X_test, cmap=cmap, show=False)
    
    import matplotlib.pyplot as pl
    pl.savefig(f'pics/{target_column}_shap.png')

    print(f"saved SHAP for{target_column}")
    
    return 
 
if __name__ == "__main__":
    print("start running")
    print("\ ---------------------------------------------------------------- \n")
    data = pd.read_csv('Dataset_normalized.csv')  
    categories = ["Generic policy", "Reporting mechanism", "Scope of practice", "User guideline"]
    
    results = []
    

    for target_column in categories:
        for feature_set_name, feature_list in feature_types.items():
            print(f"\nProcessing {target_column} with {feature_set_name} features...")

            # Ensure only selected features + target column are used
            selected_columns = feature_list + [target_column]
            data_selected = data[selected_columns].dropna()  # Drop rows with missing values

            # Train the model
            accuracy, report, auc, classifier, macro_avg_f1, X_train, X_test, y_train, y_test = model(data_selected, target_column)
            sklearn_regressor = get_model(classifier.show_models())

            print(f"The best model for {target_column} ({feature_set_name}) is {sklearn_regressor}")

            # Save results
            results.append([target_column, feature_set_name, accuracy, auc, macro_avg_f1, sklearn_regressor])

    # Output the results
    results_df = pd.DataFrame(results, columns=['Target Column', 'Accuracy', 'AUC', 'Macro Avg F1', 'Best Model'])
    results_df.to_csv('model_results.csv', index=False)
    print("All processing complete. Results saved to model_results.csv.")
    
    
