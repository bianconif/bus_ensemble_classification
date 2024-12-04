from itertools import combinations, product
import numpy as np
import pandas as pd
from tabulate import tabulate

from ml_routines.src.performance_estimation import cross_validation,\
     internal_validation

from common import clfs, scalers, testing_conditions, single_descriptors
from common import datasets_root, datasets_metadata_file, n_splits,\
     features_root_folder, train_test_split_method
from common import binary_class_labels, class_column, feature_prefix,\
     pattern_id_column
from common import acc_ci_alpha, acc_ci_method, best_res_single_file,\
     complete_res_single_file, ranking_single_file
from functions import get_feature_columns, pack_results

experimental_conditions = product(
    single_descriptors, testing_conditions, clfs, scalers
)
experimental_conditions = list(experimental_conditions)

df_results = pd.DataFrame()

for ec_idx, experimental_condition in enumerate(experimental_conditions):
    record = {'Descriptor': experimental_condition[0],
              'Train': experimental_condition[1]['train'],
              'Test': experimental_condition[1]['test'],
              'Classifier': experimental_condition[2],
              'Scaler': experimental_condition[3]}
    
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
        #Perform internal validation
        feature_src = f'{features_root_folder}/{record["Train"]}/{record["Descriptor"]}.csv'
        metadata_src = f'{datasets_root}/{record["Train"]}{datasets_metadata_file}'
        splits_file = f'{datasets_root}/{record["Train"]}/splits/splits.csv'
        
        df_features = pd.read_csv(feature_src)
        df_metadata = pd.read_csv(metadata_src)
        
        results = internal_validation(
            df_features=df_features, 
            df_metadata=df_metadata, 
            splits_file=splits_file, 
            split_method=train_test_split_method, 
            split_method_params={'n_splits': n_splits}, 
            feature_columns=get_feature_columns(df_features, 
                                                feature_prefix),
            binary_output=True, 
            binary_class_labels=binary_class_labels,
            **common_params
        )
                
        #Create result record        
        _means = np.mean(results, axis=0)
        record.update(
            pack_results(
                acc=_means[0], sens=_means[1], spec=_means[2], 
                n_test_samples=df_features.shape[0], alpha=acc_ci_alpha, 
                ci_method=acc_ci_method
            )
        )
               
    else:
        #Perform cross validation
        
        train_feature_src = f'{features_root_folder}/{record["Train"]}/{record["Descriptor"]}.csv'
        train_metadata_src = f'{datasets_root}/{record["Train"]}{datasets_metadata_file}'
        test_feature_src = f'{features_root_folder}/{record["Test"]}/{record["Descriptor"]}.csv'
        test_metadata_src = f'{datasets_root}/{record["Test"]}{datasets_metadata_file}'        
       
        df_train_features = pd.read_csv(train_feature_src)
        df_train_metadata = pd.read_csv(train_metadata_src) 
        df_test_features = pd.read_csv(test_feature_src)
        df_test_metadata = pd.read_csv(test_metadata_src)
        
        feature_columns = get_feature_columns(
            df_train_features, feature_prefix=feature_prefix
        )
        #Need to check that feature columns are the same in the train and 
        #test set
        
        results = cross_validation(
            df_train=df_train_features, 
            df_test=df_test_features, 
            df_train_metadata=df_train_metadata, 
            df_test_metadata=df_test_metadata, 
            feature_columns=feature_columns, 
            **common_params)  
        
        #Create result record        
        record.update(
            pack_results(
                acc=results['accuracy'], 
                sens=results[binary_class_labels[0]]['recall'], 
                spec=results[binary_class_labels[1]]['recall'], 
                n_test_samples=df_features.shape[0], alpha=acc_ci_alpha, 
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
    
df_results.to_csv(complete_res_single_file)
df_best_by_feature_set.to_csv(best_res_single_file)

#=========================================================
#=============== Ranking of descriptors ==================
#=========================================================
df_pruned = pd.read_csv(best_res_single_file)
subset = ['Descriptor', 'Train', 'Test']
df_pruned = df_pruned.drop_duplicates(subset=subset)
df_pruned.set_index(keys=subset, inplace=True)

df_round_robin = pd.DataFrame(
    index=df_pruned.index.get_level_values('Descriptor').unique(),
    columns=['Wins', 'Losses', 'Ties'], 
)
df_round_robin.fillna(0, inplace=True)

pairing_table = combinations(df_round_robin.index, r=2)
for home, visitor in pairing_table:
    for train, test in product(
        df_pruned.index.get_level_values('Train').unique(), 
        df_pruned.index.get_level_values('Test').unique()
    ):
        
        home_record=df_pruned.loc[home, train, test]
        visitor_record=df_pruned.loc[visitor, train, test]
        
        #Assign the points
        if home_record['Acc. CI_l'] > visitor_record['Acc. CI_u']:
            #Home wins
            df_round_robin.loc[home]['Wins'] += 1
            df_round_robin.loc[visitor]['Losses'] += 1
        elif visitor_record['Acc. CI_l'] > home_record['Acc. CI_u']:
            #Visitor wins
            df_round_robin.loc[visitor]['Wins'] += 1
            df_round_robin.loc[home]['Losses'] += 1
        else:
            #Tie
            df_round_robin.loc[visitor]['Ties'] += 1
            df_round_robin.loc[home]['Ties'] += 1  
            
df_round_robin['Points'] = (0 * df_round_robin['Losses']) +\
                           (1 * df_round_robin['Ties']) +\
                           (2 * df_round_robin['Wins'])
df_round_robin['Rank'] = df_round_robin['Points'].rank(ascending=False)
df_round_robin.sort_values(by='Rank', ascending=True, inplace=True)
df_round_robin.to_csv(ranking_single_file)
#=========================================================
#=========================================================
#=========================================================

print('Complete results')
print(tabulate(df_results, headers='keys', floatfmt="3.1f"))
print()

print('Best accuracy of single descriptors')
print(tabulate(df_best_by_feature_set, headers='keys', floatfmt="3.1f"))
print()

print('Ranking of single descriptors')
print(tabulate(df_round_robin, headers='keys', floatfmt="3.1f"))