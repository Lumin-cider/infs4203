# INFS4203 assignment 2 readme
This document contains information regarding the results of the data analysis for the second assignment of INFS4203.

It includes the final choices for preprocessing and classification methods, along with the specific hyperparameters used. Additionally, it outlines the environment in which the analysis was performed and provides instructions on how to reproduce the exact results presented in the report.
## Final Choices
The preprocessing process involves three steps: imputation, normalization, and anomaly detection. After experimenting with various methods for each step and applying them in different orders, the final preprocessing order that yielded the best performance (based on the macro F1 score) is imputation, normalization, and anomaly detection. The specific methods used are class-specific imputation, MinMax normalization, and density-based anomaly detection. For density-based anomaly detection, the hyperparameters that provided the highest F1 score are contamination = 0.48 and n_neighbors = 5.

For classification, out of the five main methods tested (KNN, Naive Bayes, Decision Tree, Random Forest, and Ensemble), Random Forest achieved the highest macro F1 score. The best-performing hyperparameters were n_estimators = 25, criterion = gini, with no max depth specified.

## Environment Description
During preprocessing and training, the work was done using scikit-learn for machine learning and pandas for data manipulation. The environment was set up on Windows as the operating system, with Python 3.11.4 as the programming language and version. Additionally, an OS environment variable called OMP_NUM_THREADS was adjusted to 7 to accommodate kMeans, as kMeans has been known to have memory leak issues on Windows when using MKL. This warning was shown during execution, so the change was added to prevent potential memory leaks
## Reproduction Instructions
To reproduce the report results, the process in the main method should automatically run the preprocessing and training procedures. The method will first perform preprocessing, which tries all possible orders of imputation and normalization, storing all processed DataFrames in a list. At the end, it returns the DataFrame with the highest F1 macro score.

Once the DataFrame is returned, its index is reset, as removing the outliers has caused some rows to be omitted, resulting in missing indices. After resetting, the DataFrame is sent to the training method, which trains multiple models. The model with the highest F1 macro score returns its classifier and scores to the main method. This classifier is then used in the predict method, where the test data is imported and normalized in the same way as the training data. After normalization, the classifier outputs predictions for the test data, which are then written into the report file.

Please note that the dbscanCluster method is implemented in the file but is not actually used. During implementation, it was observed that DBSCAN tends to classify all data points as outliers, which results in the removal of the entire dataset.
