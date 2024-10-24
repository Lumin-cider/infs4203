import os
os.environ["OMP_NUM_THREADS"] = "7"
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, LocalOutlierFactor
import scipy.stats as stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest





def cv(df):
    X_train = df.iloc[:,:-1].values
    y_train = df.iloc[:,-1].values
    kNN_5 = KNeighborsClassifier()

    f1_kNN_5 = cross_val_score(kNN_5, X_train, y_train, cv=5, scoring=('f1_macro'))
    # print("The cross-validation f1-score is: ", f1_kNN_5.mean())

    return f1_kNN_5.mean()


def imputation(df):
    df_impu_all = df.copy()
    df_impu_all.iloc[:,:103] = df_impu_all.iloc[:,:103].fillna(df_impu_all.iloc[:,:103].mean())
    df_impu_all.iloc[:,103:] = df_impu_all.iloc[:,103:].fillna(df_impu_all.iloc[:,103:].mode().iloc[0])

    df_impu_class = df.copy()
    cat_list = df_impu_class.iloc[:,-1].unique()

    for cat in cat_list:
        #imputate numerical values
        df_impu_class.loc[df_impu_class.iloc[:,-1]==cat,df_impu_class.columns[:103]] =\
            df_impu_class.loc[df_impu_class.iloc[:,-1]==cat,df_impu_class.columns[:103]].\
                fillna(df_impu_class.loc[df_impu_class.iloc[:,-1]==cat,df_impu_class.columns[:103]].mean())

        #imputate categorical values
        df_impu_class.loc[df_impu_class.iloc[:,-1]==cat,df_impu_class.columns[103:]] = \
            df_impu_class.loc[df_impu_class.iloc[:,-1]==cat,df_impu_class.columns[103:]].\
                fillna(df_impu_class.loc[df_impu_class.iloc[:,-1]==cat,df_impu_class.columns[103:]].mode().iloc[0])
    
    return df_impu_all, df_impu_class

    
def normalize(df):
    df_minmax = df.copy()
    scaler = MinMaxScaler()
    scaler.fit(df_minmax.loc[:,df_minmax.columns[:103]])
    df_minmax.loc[:,df_minmax.columns[:103]] = scaler.transform(df_minmax.loc[:,df_minmax.columns[:103]])
    

    df_standard = df.copy()
    scaler = StandardScaler()
    scaler.fit(df_standard.loc[:,df_standard.columns[:3]])
    df_standard.loc[:,df_standard.columns[:3]] = scaler.transform(df_standard.loc[:,df_standard.columns[:3]])
    
    return df_minmax, df_standard

def densityAnomaly(df):
    x = df.iloc[:,:-1].values
    max_score = 0
    max_con = 0
    max_nei = 0
    best_df = None
    for contamination in np.arange(0.01, 0.5, 0.01):
        for n_neighbors in np.arange(1, 30, 2):
            LOF = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            y_pred_LOF = LOF.fit_predict(x)
            X_scores = -LOF.negative_outlier_factor_ # High LOF: Outliers, Low LOF: Inliers.
            df_cleaned = df[y_pred_LOF == 1]
            score = cv(df_cleaned)
            if (score > max_score):
                max_score = score
                max_con = contamination
                max_nei = n_neighbors
                best_df = df_cleaned
    return best_df

def modelAnomaly(df):
    X = df.iloc[:,:-3].values
    likelihoods = np.zeros_like(X, dtype=float)
    filters = np.zeros_like(X, dtype=float)

    for col in range(X.shape[1]):
        # Calculate the mean and standard deviation for the column
        mean = np.mean(X[:, col])
        std_dev = np.std(X[:, col])
        # Calculate the PDF (likelihood) for each row in the column
        likelihoods[:, col] = stats.norm.pdf(X[:, col], loc=mean, scale=std_dev)
        filters[:, col] = stats.norm.pdf((mean+(2*std_dev)), loc=mean, scale=std_dev)

    condition_mask = np.any(likelihoods < filters, axis=1)
    df_cleaned = df[~condition_mask]
    return df_cleaned

def distAnomaly(df):
    X = df.iloc[:,:-1].values
    best = 0
    best_nei = 0
    best_thresh = 0
    best_df = None
    distance = None
    for thresh in np.arange(20, 95, 5):
        for n_neighbors in np.arange(1,30, 2):
            nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
            distances, indices = nbrs.kneighbors(X)
            distance_score = distances[:,n_neighbors-1]
            threshold = np.percentile(distance_score, thresh)
            # Boolean mask for outliers
            outlier_mask = distance_score > threshold
            # Get indices of the outlier rows
            outlier_indices = np.where(outlier_mask)[0]
            df_cleaned = df.drop(outlier_indices, axis=0).reset_index(drop=True)
            
            score = cv(df_cleaned)
            if score > best:
                best = score
                best_nei = n_neighbors
                best_thresh = thresh
                distance = distance_score
                best_df = df_cleaned
    return best_df

def kmeancluster(df):
    X = df.iloc[:, :-1].values
    best = 0
    best_thresh = 0
    best_df = None
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    cluster_labels_kmeans = kmeans.predict(X)
    distances = np.min(kmeans.transform(X), axis=1)
    average_distances = []
    for i in range(2):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        avg_distance = np.mean(distances[cluster_indices])
        average_distances.append(avg_distance)
    
    relative_distances = []
    for distance, label in zip(distances, cluster_labels_kmeans):
        # Get the average distance for the cluster of the data point
        average_distance = average_distances[label]
        # Calculate the relative distance
        relative_distance = distance / average_distance
        # Append the relative distance to the list
        relative_distances.append(relative_distance)
    for threshold in range(50, 90, 5):
        threshold_relative_distances = np.percentile(relative_distances, 90)
        anomalies_relative_distances = relative_distances > threshold_relative_distances
        df_cleaned = df.loc[~anomalies_relative_distances,:]
        score = cv(df_cleaned)
        if score > best:
            best = score
            best_thresh = threshold
            best_df = df_cleaned
    return best_df

def dbscancluster(df):
    X = df.iloc[:, :-1].values
    best = 0
    best_sample = 0
    best_eps = 0
    best_df = None
    for sample in range(2, 10, 1):
        for epsilon in np.arange(0.5, 10, 0.5):
            dbscan = DBSCAN(eps=epsilon, min_samples = sample)
            dbscan.fit(X)
            cluster_labels_DBSCAN = dbscan.labels_

            df_cleaned = df.loc[cluster_labels_DBSCAN == -1]
            score = cv(df_cleaned)
            if score > best:
                best = score
                best_eps = epsilon
                best_sample = sample
                best_df = df_cleaned
    return best_df

def isoforest(df):
    X = df.iloc[:, :-1].values
    IF = IsolationForest(random_state=0)
    IF.fit(X)
    # Perform the prediction using the IsolationForest
    outlier_labels = IF.predict(X)

    # Create a mask for inliers (rows that are not outliers)
    inlier_mask = outlier_labels != -1

    # Apply the mask to the original DataFrame to keep only the inlier rows
    df_cleaned = df[inlier_mask]
    return df_cleaned

def preprocessing():
    out_detect = [("kmeancluster", kmeancluster), ("densityAnomaly", densityAnomaly), ("modelAnomaly", modelAnomaly), ("distAnomaly", distAnomaly), ("isoforest", isoforest)]
    original_df = pd.read_csv("DM_Project_24.csv")
    df_all, df_class = imputation(original_df)
    dfs = [];

    for imp_type, df in (("imp_all", df_all), ("imp_class", df_class)):
        minmax, standard = normalize(df)
        for anom_type, method in out_detect:
            # outlier detection first
            processed = method(df)
            # normalization
            processed_minmax, processed_standard = normalize(processed)
            dfs.append(((imp_type, anom_type, "minmax"), processed_minmax))
            dfs.append(((imp_type, anom_type, "standard"), processed_standard))

            # outlier detection
            dfs.append(((imp_type, "minmax", anom_type),method(minmax)))
            dfs.append(((imp_type, "standard", anom_type), method(standard)))

    best_df = None
    best_score = 0
    best_order = None

    for order, df in dfs:
        score = cv(df)
        if score > best_score:
            best_score = score
            best_df = df
            best_order = order

    # imp class -> minmax -> density Anomaly (contamination = 0.48, n_neighbour = 5) -> 0.934
    print(best_order, best_score)
    return best_df

def knn_train(X_train, y_train):
    parameters = [{'n_neighbors': [int(x) for x in np.arange(1, 22, 2)]}]
    kNN = KNeighborsClassifier()
    clf_best_kNN = GridSearchCV(kNN, parameters, cv=5, scoring='f1_macro')
    clf_best_kNN.fit(X_train, y_train)

    print("knn best parameters: ", clf_best_kNN.best_params_, " with f1-macro score of ", clf_best_kNN.best_score_)
    return clf_best_kNN.best_score_, clf_best_kNN.best_estimator_


def ohe_transform(df):
    ohe = OneHotEncoder()
    feature_array = ohe.fit_transform(df.iloc[:,103:105]).toarray()
    
    feature_labels = []
    for i, categories in zip(range(103, 105), ohe.categories_):
        for cat in categories:
            feature_labels.append(f"{df.columns[i]}_{int(cat)}")

    features = pd.DataFrame(feature_array, columns = feature_labels)
    df_new = pd.concat([df.iloc[:,:103],features,df.iloc[:,-1]],axis = 1)
    
    return df_new

def tree_train(X_train, y_train):
    dt = DecisionTreeClassifier(random_state = 0)
    # Define the hyperparameter grid to search
    param_grid = {
        'criterion': ['gini', 'entropy'],  # Split criterion
        'max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],   # Depth of the tree
    }

    # Initialize GridSearchCV
    clf_best_dt = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='f1_macro')
    clf_best_dt.fit(X_train, y_train)

    print("Decision Tree best parameters: ", clf_best_dt.best_params_, " with f1-macro score of ", clf_best_dt.best_score_)
    return clf_best_dt.best_score_, clf_best_dt.best_estimator_
    

def forest_train(X_train, y_train):
    rf = RandomForestClassifier(random_state=0)

    # Define the hyperparameter grid to search
    param_grid = {
        'n_estimators': range(10, 50, 5),  # Number of trees in the forest
        'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
        'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    }

    # Initialize GridSearchCV
    clf_best_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1_macro')
    clf_best_rf.fit(X_train, y_train)

    print("Random Forest best parameters: ", clf_best_rf.best_params_, " with f1-macro score of ", clf_best_rf.best_score_)
    return clf_best_rf.best_score_, clf_best_rf.best_estimator_

def training(df):
    df = ohe_transform(df)
    X_train = df.iloc[:,:-1]
    y_train = df.iloc[:,-1]

    classifiers = []
    knn = knn_train(X_train, y_train)
    classifiers.append(knn)

    gnb = GaussianNB()
    f1_gnb = cross_val_score(gnb, X_train, y_train, cv=5, scoring=('f1_macro'))
    print("Naive Bayes has an f1-macro score of: ", f1_gnb.mean())

    gnb.fit(X_train, y_train)
    classifiers.append((f1_gnb.mean(), gnb))

    tree = tree_train(X_train, y_train)
    classifiers.append(tree)

    forest = forest_train(X_train, y_train)
    classifiers.append(forest)

    knn_vote = KNeighborsClassifier(**(knn[1].get_params()))
    gnb_vote = GaussianNB()
    forest_vote = RandomForestClassifier(**(forest[1].get_params()))
    voting_clf = VotingClassifier(estimators=[('knn', knn_vote), ('naive', gnb_vote), ('forest', forest_vote)], voting='soft')
    f1_voting = cross_val_score(voting_clf, X_train, y_train, cv=5, scoring=('f1_macro'))
    print("Ensemble has an f1-macro score of: ", f1_voting.mean())
    
    voting_clf.fit(X_train, y_train)
    classifiers.append((f1_voting.mean(), voting_clf))

    best_f1_macro, best_classifier = max(classifiers, key=lambda x: x[0])
    print("Best classifier:", best_classifier)
    print("Best f1_macro:", best_f1_macro)

    new_best = type(best_classifier)(**(best_classifier.get_params()))
    accu_calc = cross_val_score(new_best, X_train, y_train, cv=5, scoring=('accuracy'))
    return accu_calc.mean(), best_f1_macro, best_classifier

def predict(classifier):
    df = pd.read_csv("DM_Project_24.csv")
    imp_all, imp_class = imputation(df)
    scaler = MinMaxScaler()
    scaler.fit(imp_class.loc[:,imp_class.columns[:103]])
    

    test = pd.read_csv("test_data(1).csv")
    test.loc[:,test.columns[:103]] = scaler.transform(test.loc[:,test.columns[:103]])

    test = ohe_transform(test)
    test = test.iloc[:,:-1]
    prediction = classifier.predict(test)
    with open("s4650048.infs4203", "w") as file:
        for pred in prediction:
            file.write(f"{pred},\n")





def main():
    processed_df = preprocessing()
    processed_df = processed_df.reset_index(drop=True)
    best_accuracy, best_f1_macro, best_classifier = training(processed_df)
    predict(best_classifier)
    with open("s4650048.infs4203", "a") as file:
            file.write(f"{best_accuracy:.3f},{best_f1_macro:.3f}\n")


if __name__ == "__main__":
    main()





