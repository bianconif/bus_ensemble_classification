
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from ml_routines.src.performance_estimation import cross_validation,\
     internal_validation

from common import class_column, feature_prefix, n_splits,\
     pattern_id_column, train_test_split_method
from functions import get_feature_columns
 
df_train = pd.read_csv('features/BrEaST/Gabor.csv')
df_test = pd.read_csv('features/BUID/Gabor.csv')

df_train_metadata = pd.read_csv('../data/datasets/BrEaST/metadata/metadata.csv')
df_test_metadata = pd.read_csv('../data/datasets/BUID/metadata/metadata.csv')

clf = SVC(probability=True)
scaler = StandardScaler()

common_params = {'clf': clf,
                 'scaler': scaler, 
                 'method': train_test_split_method, 
                 'method_params': {'n_splits': n_splits},
                 'binary_output': True,
                 'binary_class_labels': ['1', '0']}

#Internal validation
#valid_report_1 = internal_validation(
    #df_features=df_train, df_metadata=df_train_metadata, 
    #splits_file='../data/datasets/BrEaST/splits/splits.csv',
    #**common_params
#)
#valid_report_2 = internal_validation(
    #df_features=df_test, df_metadata=df_test_metadata,
    #splits_file='../data/datasets/BUID/splits/splits.csv',
    #**common_params
#)    

classification_report, df_complete_report = cross_validation(
    df_train, df_test, df_train_metadata, df_test_metadata, clf, scaler, 
    pattern_id_column, class_column, 
    get_feature_columns(df_train, feature_prefix),
    complete_report=True)
a = 0
