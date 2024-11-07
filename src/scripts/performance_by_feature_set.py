from itertools import product
import numpy as np
import pandas as pd
from tabulate import tabulate

from ml_routines.src.performance_estimation import cross_validation,\
     _get_sensitivity_specificity, internal_validation

from common import clfs, scalers, testing_conditions, descriptors
from common import datasets_root, datasets_metadata_file, n_splits,\
     features_root_folder, train_test_split_method
from common import binary_class_labels, class_column, feature_prefix,\
     pattern_id_column
from functions import get_feature_columns

experimental_conditions = product(
    descriptors, testing_conditions, clfs, scalers
)

df_results = pd.DataFrame()

for experimental_condition in experimental_conditions:
    record = {'Descriptor': experimental_condition[0],
              'Train': experimental_condition[1]['train'],
              'Test': experimental_condition[1]['test'],
              'Classifier': experimental_condition[2],
              'Scaler': experimental_condition[3]}
    
    print(f'Testing {record}')
    
    common_params = {'clf': clfs[record['Classifier']],
                     'scaler': scalers[record['Scaler']],
                     'pattern_id_column': pattern_id_column,
                     'class_column': class_column
                     }
    
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
            method=train_test_split_method, 
            method_params={'n_splits': n_splits}, 
            feature_columns=get_feature_columns(df_features, 
                                                feature_prefix),
            binary_output=True, 
            binary_class_labels=binary_class_labels,
            **common_params
        )
        
        #Convert to %
        results = 100*results
        
        _means, _stds = np.mean(results, axis=0), np.std(results, axis=0)
        record.update({'Acc. avg': _means[0], 'Acc. std': _stds[0],
                       'Sens.': _means[1], 'Spec.': _means[2]})         
        
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
        
        acc = results['accuracy']
        sens, spec = _get_sensitivity_specificity(
            results, binary_class_labels=binary_class_labels
        )  
        
        #Convert to %
        record.update({'Acc. avg': 100*acc, 'Acc. std': None,
                       'Sens.': 100*sens, 'Spec.': 100*spec})         
      
            
    df_results = pd.concat((df_results, 
                            pd.DataFrame(data=record, index=[0])))
    
df_results.to_csv('performace-by-feature-set.csv')
    
print(tabulate(df_results, headers='keys', floatfmt="3.1f"))