import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import f1_score
from sklearn.neighbors import LocalOutlierFactor
import scipy.stats as stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest


def cv(data):
    kf = KFold(n_splits=5, shuffle = True, random_state=0)
    
    f1s = []

    for train, test in kf.split(data.index):
        knn = KNeighborsClassifier(n_neighbors=3)
        train_set = data.iloc[train]
        test_set = data.iloc[test]
        knn.fit(train_set.iloc[:, :105], train_set.iloc[:, -1])

        pred = knn.predict(test_set.iloc[:, :105])
        f1 = f1_score(test_set.iloc[:, -1], pred, zero_division=1)
        f1s.append(f1)

    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)

    # print("Macro f1:", mean_f1)
    # print("Standard deviation of f1:", std_f1)
    # print('----------------------------------------------------------')
    return mean_f1


def imputation(df):
    df_impu_all = df.copy()
    df_impu_all.iloc[:,:103] = df_impu_all.iloc[:,:103].fillna(df_impu_all.iloc[:,:103].mean())
    df_impu_all.iloc[:,103:] = df_impu_all.iloc[:,103:].fillna(df_impu_all.iloc[:,103:].mode().iloc[0])
    df_impu_all.to_csv("df_impu_all.csv", index=False)

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

    df_impu_class.to_csv("df_impu_class.csv", index=False)

    
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
            # print(f'LOF Anomaly Detection with contamination={contamination:.2f} and k={n_neighbors}')
            LOF = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            y_pred_LOF = LOF.fit_predict(x)
            X_scores = -LOF.negative_outlier_factor_ # High LOF: Outliers, Low LOF: Inliers.
            df_cleaned = df[y_pred_LOF == 1]
            score = cv(df_cleaned)
            if ( score > max_score):
                max_score = score
                max_con = contamination
                max_nei = n_neighbors
                best_df = df_cleaned

    print(max_nei, max_con, max_score)
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
    df_cleaned.to_csv("result.csv")
    print(cv(df_cleaned))
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
    print(best, best_nei, best_thresh, distance)
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
    print(best, best_thresh)
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
    print(best, best_sample, best_eps)
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
    print(cv(df_cleaned))
    return df_cleaned


out_detect = [densityAnomaly, modelAnomaly, distAnomaly, kmeancluster, dbscancluster, isoforest]
df_all = pd.read_csv("df_impu_all.csv")
df_class = pd.read_csv("df_impu_class.csv")
dfs = [];

dbscancluster(df_class)


# for df in (df_all, df_class):
#     minmax, standard = normalize(df)
#     for method in out_detect:
#         # outlier detection first
#         processed = method(df)
#         # normalization
#         processed_minmax, processed_standard = normalize(processed)
#         dfs.append(processed_minmax)
#         dfs.append(processed_standard)

#         # outlier detection
#         dfs.append(method(minmax))
#         dfs.append(method(standard))

# best_df = None
# best_score = 0;

# for df in dfs:
#     score = cv(df)
#     if score > best_score:
#         best_score = score
#         best_df = df

# print(best_score)
# pd.DataFrame(best_df).to_csv("best_df.csv", index=False)


    






