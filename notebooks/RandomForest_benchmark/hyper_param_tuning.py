# Script to run hyperparameter optimisation on the Random Forest model.
# This was found to have negligible impact on model performance,
# but is included for completeness.

import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
from hm4.labelling import CLASS_NAMES

# data directory for the data preprocessed as by Dhoundiyal et al. (2025) and aggregated.
# See scripts/dhoundiyal_benchmark for the preprocessing scripts.
TRAIN_DATA_PATH = #
TEST_DATA_PATH = #
xy_train = pd.read_json(TRAIN_DATA_PATH)
xy_test = pd.read_json(TEST_DATA_PATH)


x_train = xy_train.iloc[:, :-3].to_numpy()
x_test = xy_test.iloc[:, :-3].to_numpy()

y_train = xy_train["Pixel_Class"].to_numpy()
y_test = xy_test["Pixel_Class"].to_numpy()

# Define the parameter grid
param_dist = {
    "n_estimators": [512, 1024, 1536, 2048],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 10, 20, 30],
}

rf = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=10,
    scoring="accuracy",
    cv=cv,
    random_state=42,
    verbose=1,
    n_jobs=-1,
)

random_search.fit(x_train, y_train)

print(f"Best Parameters from Tuning:{random_search.best_params_}")
print("Best Accuracy Score on Cross-Val: {random_search.best_score_}")

y_pred = random_search.predict(x_test)

true_unique_classes = np.unique(y_test)
predicted_unique_classes = np.unique(y_pred)
classes_to_calculate_f1 = np.unique(
    np.concatenate((true_unique_classes, predicted_unique_classes))
)
string_labels = np.array(
    [CLASS_NAMES[int(obs_class)] for obs_class in classes_to_calculate_f1]
)

report = classification_report(y_test, y_pred, output_dict=True)
print(f"Accuracy: {report['accuracy']}")
print(f"Macro F1: {report['macro avg']['f1-score']}")
print(f"Weighted F1: {report['weighted avg']['f1-score']}")
print(f"Precision: {report['macro avg']['precision']}")
print(f"Recall: {report['macro avg']['recall']}")

cm = confusion_matrix(y_test, y_pred, normalize="true")
fig, ax = plt.subplots(figsize=(14, 14))
ax.matshow(cm, cmap="viridis")
ax.set_xticks(range(len(string_labels)))
ax.set_yticks(range(len(string_labels)))
ax.set_xticklabels(string_labels, rotation=90)
ax.set_yticklabels(string_labels)
for i in range(len(string_labels)):
    for j in range(len(string_labels)):
        if i != j:
            continue
        ax.text(
            j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="black"
        )
plt.savefig("Hyperparam_tuned_confusion_matrix.png", dpi=300)
