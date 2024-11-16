import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml_routines.src.combining import late_fusion

from common import class_column, feature_prefix, pattern_id_column,\
     pattern_id_column
from functions import get_feature_columns

clf = SVC(probability=True)
scaler = StandardScaler()

df_train_dcf = pd.read_csv('features/BrEaST/DCF.csv')
df_train_gabor = pd.read_csv('features/BrEaST/Gabor.csv')
df_train_morpho = pd.read_csv('features/BrEaST/Morphological.csv')

df_test_dcf = pd.read_csv('features/BUID/DCF.csv')
df_test_gabor = pd.read_csv('features/BUID/Gabor.csv')
df_test_morpho = pd.read_csv('features/BUID/Morphological.csv')

df_train_metadata = pd.read_csv('../data/datasets/BrEaST/metadata/metadata.csv')
df_test_metadata = pd.read_csv('../data/datasets/BUID/metadata/metadata.csv')

dfs_train = [df_train_dcf, df_train_gabor, df_train_morpho]
dfs_test = [df_test_dcf, df_test_gabor, df_test_morpho]

feature_columns = list()
for df_train in dfs_train:
    feature_columns.append(
        get_feature_columns(df_train, feature_prefix)
    )


late_fusion(dfs_train, dfs_test, df_train_metadata, 
            df_test_metadata, clf, scaler,
            pattern_id_column, class_column, feature_columns,
            fusion_method='majority-voting')

feature_columns = list()
for df_features in dfs_features:
    feature_columns.append(get_feature_columns(
        df_features, feature_prefix))

