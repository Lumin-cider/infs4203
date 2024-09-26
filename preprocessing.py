import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

# df = pd.read_csv("DM_Project_24.csv")

# df_impu_all = df.copy()

# df_impu_all.iloc[:,:103] = df_impu_all.iloc[:,:103].fillna(df_impu_all.iloc[:,:103].mean())
# df_impu_all.iloc[:,103:] = df_impu_all.iloc[:,103:].fillna(df_impu_all.iloc[:,103:].mode().iloc[0])
# df_impu_all.to_csv("df_impu_all.csv", index=False)

# df_impu_class = df.copy()
# cat_list = df_impu_class.iloc[:,-1].unique()

# for cat in cat_list:
#     #imputate numerical values
#     df_impu_class.loc[df_impu_class.iloc[:,-1]==cat,df_impu_class.columns[:103]] =\
#         df_impu_class.loc[df_impu_class.iloc[:,-1]==cat,df_impu_class.columns[:103]].\
#             fillna(df_impu_class.loc[df_impu_class.iloc[:,-1]==cat,df_impu_class.columns[:103]].mean())

#     #imputate categorical values
#     df_impu_class.loc[df_impu_class.iloc[:,-1]==cat,df_impu_class.columns[103:]] = \
#         df_impu_class.loc[df_impu_class.iloc[:,-1]==cat,df_impu_class.columns[103:]].\
#             fillna(df_impu_class.loc[df_impu_class.iloc[:,-1]==cat,df_impu_class.columns[103:]].mode().iloc[0])

# df_impu_class.to_csv("df_impu_class.csv", index=False)

def cv(data):
    kf = KFold(n_splits=10, shuffle = True, random_state=0)
    
    f1s = []

    for train, test in kf.split(data.index):
        knn = KNeighborsClassifier(n_neighbors=1)
        train_set = data.iloc[train]
        test_set = data.iloc[test]
        knn.fit(train_set.iloc[:, :105], train_set.iloc[:, -1])

        pred = knn.predict(test_set.iloc[:, :105])
        f1 = f1_score(test_set.iloc[:, -1], pred)
        f1s.append(f1)

    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)

    print("Macro f1:", mean_f1)
    print("Standard deviation of f1:", std_f1)
    print('----------------------------------------------------------')

df_impu_all = pd.read_csv("df_impu_all.csv")
cv(df_impu_all)
df_impu_class = pd.read_csv("df_impu_class.csv")
cv(df_impu_class)

