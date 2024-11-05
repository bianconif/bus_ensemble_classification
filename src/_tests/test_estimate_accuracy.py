
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from ml_routines.src.performance_estimation import estimate_accuracy,\
     estimate_accuracy_k_fold
 
df_train = pd.read_csv('features/BrEaST/Gabor.csv')
df_test = pd.read_csv('features/BUID/Gabor.csv')

df_train_metadata = pd.read_csv('../data/datasets/BrEaST/metadata/metadata.csv')
df_test_metadata = pd.read_csv('../data/datasets/BUID/metadata/metadata.csv')

clf = LinearSVC()
scaler = StandardScaler()

#Internal validation
acc, sensitivity, specificity =estimate_accuracy_k_fold(
    df_features=df_train, df_metadata=df_train_metadata, clf=clf, 
    scaler=scaler, n_folds=5, splits_file='../data/datasets/BrEaST/splits/splits.csv')
acc, sensitivity, specificity =estimate_accuracy_k_fold(
    df_features=df_test, df_metadata=df_test_metadata, clf=clf,
    scaler=scaler, n_folds=5,
    splits_file='../data/datasets/BUID/splits/splits.csv')    

_, _acc, sens, spec = estimate_accuracy(df_train, df_test, 
                                         df_train_metadata, 
                                         df_test_metadata, 
                                         clf, scaler)
a = 0
