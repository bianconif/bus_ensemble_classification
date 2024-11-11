import pandas as pd

from ml_routines.src.combining import concatenate_features

from common import feature_prefix, pattern_id_column
from functions import get_feature_columns

df_dcf = pd.read_csv('features/BrEaST/DCF.csv')
df_gabor = pd.read_csv('features/BrEaST/Gabor.csv')
df_morpho = pd.read_csv('features/BrEaST/Morphological.csv')

dfs_features = [df_dcf, df_gabor, df_morpho]
feature_columns = list()
for df_features in dfs_features:
    feature_columns.append(get_feature_columns(
        df_features, feature_prefix))

concatenated_features, new_feature_columns = concatenate_features(
    dfs_features, feature_columns, pattern_id_column)
    
a = 0

