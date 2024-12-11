from itertools import product
import numpy as np
import pandas as pd
from tabulate import tabulate

from ml_routines.src.performance_estimation import\
     cross_validation_combined, internal_validation_combined
from ml_routines.src.combining import concatenate_features

from common import clfs, scalers, testing_conditions, fusion_methods,\
     combined_descriptors, complete_res_comb_file, best_res_combined_file
from common import datasets_root, datasets_metadata_file, n_splits,\
     features_root_folder, train_test_split_method
from common import binary_class_labels, class_column, feature_prefix,\
     pattern_id_column
from common import acc_ci_alpha, acc_ci_method
from functions import get_feature_columns, pack_results

experimental_conditions = product(
    combined_descriptors, fusion_methods, testing_conditions, clfs, 
    scalers
)
experimental_conditions = list(experimental_conditions)

df_results = pd.DataFrame()

for ec_idx, experimental_condition in enumerate(experimental_conditions):
    record = {'Descriptor': experimental_condition[0].name,
              'Fusion method': experimental_condition[1],
              'Train': experimental_condition[2]['train'],
              'Test': experimental_condition[2]['test'],
              'Classifier': experimental_condition[3],
              'Scaler': experimental_condition[4]}
    
    print(f'Testing {ec_idx + 1} of {len(experimental_conditions)} '
          f'{record}')
    
    common_params = {'clf': clfs[record['Classifier']].classifier,
                     'scaler': scalers[record['Scaler']],
                     'pattern_id_column': pattern_id_column,
                     'class_column': class_column
                     }
    
    #Perform hyperparameter tuning if requested
    if clfs[record['Classifier']].param_grid is not None:
        common_params.update({'param_grid': 
                             clfs[record['Classifier']].param_grid})
    
    if record['Train'] == record['Test']:
        
        #The feature files
        feature_srcs = [
            f'{features_root_folder}/{record["Train"]}/{src}.csv' for 
            src in experimental_condition[0].descriptors]
        
        #The metadata and splits file
        metadata_src = (f'{datasets_root}/{record["Train"]}'
                        f'{datasets_metadata_file}')
        splits_file = (f'{datasets_root}/{record["Train"]}/splits/'
                       f'splits.csv') 
        
        dfs_features = [pd.read_csv(feature_src) for feature_src in
                        feature_srcs]
        feature_columns_list = [get_feature_columns(
            df_features, feature_prefix) for df_features in dfs_features]
        
        df_metadata = pd.read_csv(metadata_src)        
        
        results = internal_validation_combined(
            dfs_features=dfs_features, 
            df_metadata=df_metadata,  
            fusion_method=experimental_condition[1],
            splits_file=splits_file, 
            split_method=train_test_split_method,
            split_method_params={'n_splits': n_splits}, 
            feature_columns_list=feature_columns_list,
            binary_output=True, 
            binary_class_labels=binary_class_labels,
            weights=experimental_condition[0].weights,
            **common_params
        )
        
        #Create result record        
        _means = np.mean(results, axis=0)
        record.update(
            pack_results(
                acc=_means[0], sens=_means[1], spec=_means[2], 
                n_test_samples=dfs_features[0].shape[0], 
                alpha=acc_ci_alpha, 
                ci_method=acc_ci_method
            )
        )       
           
    else:
        #Perform cross validation
        
        train_feature_srcs = [
            f'{features_root_folder}/{record["Train"]}/{src}.csv' for 
            src in experimental_condition[0].descriptors]
        test_feature_srcs = [
            f'{features_root_folder}/{record["Test"]}/{src}.csv' for 
            src in experimental_condition[0].descriptors]
        
        train_metadata_src = (f'{datasets_root}/{record["Train"]}'
                              f'{datasets_metadata_file}')
        test_metadata_src = (f'{datasets_root}/{record["Test"]}'
                             f'{datasets_metadata_file}')        
       
        
        df_train_metadata = pd.read_csv(train_metadata_src) 
        df_test_metadata = pd.read_csv(test_metadata_src)
                
        #Need to check that feature columns are the same in the train and 
        #test set
        
        dfs_train_features = [
            pd.read_csv(feature_src) for feature_src in 
            train_feature_srcs
        ]
        dfs_test_features = [
            pd.read_csv(feature_src) for feature_src in 
            test_feature_srcs
        ]                
        feature_columns_list = [get_feature_columns(
            df_features, feature_prefix) for df_features in 
                                dfs_train_features
        ]        
        
 
        results = cross_validation_combined(
            dfs_train=dfs_train_features, 
            dfs_test=dfs_test_features, 
            df_train_metadata=df_train_metadata,
            df_test_metadata=df_test_metadata,
            feature_columns_list=feature_columns_list, 
            fusion_method=experimental_condition[1], 
            weights=experimental_condition[0].weights,
            **common_params)            
                 
        #Create result record        
        record.update(
            pack_results(
                acc=results['accuracy'], 
                sens=results[binary_class_labels[0]]['recall'], 
                spec=results[binary_class_labels[1]]['recall'], 
                n_test_samples=dfs_test_features[0].shape[0], 
                alpha=acc_ci_alpha, 
                ci_method=acc_ci_method
            )
        )        
                
        
    df_results = pd.concat((df_results, 
                            pd.DataFrame(data=record, index=[ec_idx])))
    
#Get the best avg accuracy by feature set, train and test dataset
df_best_by_feature_set = pd.DataFrame()
for name, grp in df_results.groupby(by=['Descriptor', 'Train', 'Test']):
        
    #Index of every row where the value of 'Acc. acg' is equal to the 
    #maximum
    max_acc_idxs = grp[grp['Acc.'] == grp['Acc.'].max()].index
    df_best_by_feature_set = pd.concat((df_best_by_feature_set,
                                            df_results.loc[max_acc_idxs]),
                                           )
        
df_results.to_csv(complete_res_comb_file)
df_best_by_feature_set.to_csv(best_res_combined_file)
    
print('Complete results')
print(tabulate(df_results, headers='keys', floatfmt="3.1f"))
print()
    
print('Best accuracy of combined descriptors')
print(tabulate(df_best_by_feature_set, headers='keys', floatfmt="3.1f"))        