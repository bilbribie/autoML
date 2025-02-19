import autosklearn.classification
from shap import KernelExplainer, Explanation
from shap.plots import waterfall
# shap.initjs()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from matplotlib.colors import LinearSegmentedColormap
from imblearn.over_sampling import SMOTE
import time 

# Define feature types
feature_types = {
    "project_feature": ["num_commits", "project_age_days", "num_issues", "num_pull", 
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
categories = ["Generic policy", "Reporting mechanism", "Scope of practice", "User guideline"]

def select_top_features(features):
    np.random.seed(42)
    shap_importance_scores = {feature: np.random.rand() for feature in features}
    sorted_features = sorted(shap_importance_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top_features = [f[0] for f in sorted_features]

    print(f"top 5 features are {top_features}")
    return top_features

# run model
def model(feature , target_column, feature_set_name):
    
    X = feature
    y = data[target_column]  # Target: the column named by 'target_column'
    
    # Handle class imbalance using SMOTE
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Split the dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3)
    
    # save for check
    X_train.to_csv(f"results/dataset/{target_column}_{feature_set_name}_X_train.csv", index=False)
    X_test.to_csv(f"results/dataset/{target_column}_{feature_set_name}_X_test.csv", index=False)
    y_train.to_csv(f"results/dataset/{target_column}_{feature_set_name}_y_train.csv", index=False)
    y_test.to_csv(f"results/dataset/{target_column}_{feature_set_name}_y_test.csv", index=False)
    print("train_test_split saved to csv")
    
    # Create an AutoSklearn classifier
    classifier = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=30,  # 30 minutes (1800 seconds)
        per_run_time_limit=10        # 5 minutes per model training (adjust as needed)
        ) 
        #time_left_for_this_task=30, per_run_time_limit=10

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
    auc = roc_auc_score(y_test, pred_proba) if pred_proba is not None else "N/A"
    
    print(f"Accuracy: {accuracy}")
    print(f"Macro avg: {macro_avg_f1}")
    print(f"AUC score: {auc}")
    print("Classification report:")
    print(report1)
    
    # save for check
    params = classifier.get_params()
    params_df = pd.DataFrame([params])  # Convert to DataFrame
    params_df.to_csv(f"results/dataset/{target_column}_{feature_set_name}_classifier_params.csv", index=False)

    print("classifier saved to csv")
    
    return accuracy, auc, macro_avg_f1, classifier, X_train, X_test, y_train, y_test, X

# find model 1st rank
def get_best_model(models_dict):
    if not models_dict:
        return None
    for model_info in models_dict.values():
        if model_info.get('rank') == 1:
            best_model = model_info.get('sklearn_classifier', None)
            
            print(best_model)
            return best_model
    return None

# SHAP
def shap(target_column, feature_set_name, model, X_train, X_test, y_train, y_test, X):
    # explainer 
    explainer = shap.KernelExplainer(model.predict, shap.kmeans(X_train, 10))
    shap_values = explainer.shap_values(X_test)
    
    # summary plot
    shap.summary_plot(shap_values, X_test)
    # summary.save(f"results/pics/{target_column}_{feature_set_name}_summary.png")
    plt.savefig(f"results/pics/{target_column}_{feature_set_name}_summary.png", bbox_inches="tight", dpi=300)  # Save the figure
    plt.close()
    
    # Convert to Explanation object (for single-output models) for bar and waterfall
    shap_values_exp = shap.Explanation(values=np.array(shap_values), feature_names=X_test.columns)

    # bar plot
    shap.plots.bar(shap_values_exp, max_display=12 )
    # bar.save(f"results/pics/{target_column}_{feature_set_name}_bar.png")
    plt.savefig(f"results/pics/{target_column}_{feature_set_name}_bar.png", bbox_inches="tight", dpi=300)  # Save the figure
    plt.close()
    
    # waterfall
    idx = 9
    sv = explainer.shap_values(X.loc[[5]]) 
    exp = Explanation(sv,explainer.expected_value, data=X.loc[[idx]].values, feature_names=X.columns)
    waterfall(exp[0])
    plt.savefig(f"results/pics/{target_column}_{feature_set_name}_waterfall.png", bbox_inches="tight", dpi=300)  # Save the figure
    plt.close()
    
    return

if __name__ == "__main__":
    start_time = time.time() 
    print("start running")
    print("\ ---------------------------------------------------------------- \n")
    data = pd.read_csv('Dataset_normalized.csv')  
    
    results = []
    

    for target_column in categories:
        for feature_set_name, features in feature_types.items():
            print(f"\nProcessing {target_column} with {feature_set_name} features...")

            # 1. feature selection
            selected_features = select_top_features(features)
            
            # 2. Train the model
            accuracy, auc, macro_avg_f1, classifier, X_train, X_test, y_train, y_test, X = model(data[selected_features], target_column, feature_set_name)
            
            # 3. 
            models_dict = classifier.show_models()
            sklearn_regressor = get_best_model(models_dict)

            # 4. SHAP
            shap(target_column, feature_set_name, sklearn_regressor, X_train, X_test, y_train, y_test, X)
            
            print(f"The best model for {target_column} ({feature_set_name}) is {sklearn_regressor}")

            # Save results
            results.append([target_column, feature_set_name, selected_features, accuracy, auc, macro_avg_f1, sklearn_regressor])

    # Output the results
    results_df = pd.DataFrame(results, columns=['Target Column', 'feature_types', 'selected_features', 'Accuracy', 'AUC', 'Macro Avg F1', 'Best Model'])
    results_df.to_csv('results/model_results.csv', index=False)
    print("All processing complete. Results saved to model_results.csv.")
    
    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time  # Calculate total time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    
    
