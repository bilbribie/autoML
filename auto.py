# auto-sklearn
import autosklearn.classification
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import subprocess

# Function to evaluate the model
def evaluate_model(X_train, X_test, y_train, y_test, model):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_proba)
    return f1, auc

# Main function to run auto-sklearn multiple times and calculate averages
def run_model_selection(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    f1_scores = []
    auc_scores = []

    for _ in range(100):  # Run 100 iterations
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=120,  # Adjust time per task as necessary
            per_run_time_limit=30,
            ensemble_size=1,  # Use a single model for simplicity in analysis
            n_jobs=-1
        )
        automl.fit(X_train, y_train)
        
        f1, auc = evaluate_model(X_train, X_test, y_train, y_test, automl)
        f1_scores.append(f1)
        auc_scores.append(auc)

    # Calculate average scores
    avg_f1 = np.mean(f1_scores)
    avg_auc = np.mean(auc_scores)

    # SHAP Value Calculation
    best_model = automl.show_models().values[0]['model']  # Assuming the best model can be retrieved like this
    explainer = shap.KernelExplainer(best_model.predict_proba, X_train)
    shap_values = explainer.shap_values(X_train)

    # Save SHAP values plot
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    plt.savefig('/pics/shap_values.png')
    
    return avg_f1, avg_auc

# Git operations
def git_operations():
    try:
        subprocess.run(["git", "pull", "origin", "main"], check=True)
        subprocess.run(["git", "add", "."], check=True)
        commit_message = "Updated model"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)

        print("อัพแล้วจ้า")

    except subprocess.CalledProcessError as e:
        print(f"❌ Git operation failed: {e}")

# Example usage
if __name__ == "__main__":
    print("hello")

    git_operations()  # Perform git operations
