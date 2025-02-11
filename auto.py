#auto sklearn
!git pull origin main

import autosklearn.classification
import shap
import pandas as pd
import matplotlib.pyplot as plt

#model
print("hello")

# Push updates to GitHub
!git add .
!git commit -m "Updated model evaluations and SHAP value calculations"
!git push origin main
