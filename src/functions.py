import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from skimage.restoration import denoise_nl_means, estimate_sigma
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score

from cenotaph.basics.base_classes import Image

def denoise_nlm(img_in, patch_size=5, patch_distance=6):
    """Performs image denoising by non-linear means
    
    Parameters
    ----------
    img_in: 2D array of int (H,W)
        The gray-scale input image to be denoised.
    patch_size : int
        Size of patches used for denoising
    patch_distance: int
        Maximal distance in pixels where to search patches used for 
        denoising
    
    Returns
    -------
    img_out: 2D array of int (H,W)
        The grey-scale image after denoising
    """   
    
    img = img_in.copy()
    
    #Estimate the noise standard deviation from the noisy image
    sigma_est = np.mean(estimate_sigma(img, channel_axis=-1))
    
    #Promote image dimension to avoid denoise_nl_means_error
    img = np.expand_dims(img, axis=-1)
        
    #Perform the denoising
    img_out = denoise_nl_means(
        img, h=0.8 * sigma_est, sigma=sigma_est, fast_mode=True, 
        channel_axis=-1, preserve_range=True, patch_size=patch_size,
        patch_distance=patch_distance
    ).astype(int)  
    
    #Demote image dimension
    img = np.squeeze(img)
    
    return img_out

def compute_features(name, src_images, case_ids, dst_folder, 
                     feature_extractor, overwrite=False, verbose=True):
    """Computes features. Generates a name.csv file in the dst_folder 
    where each row corresponds to one image. Columns are organised as 
    follows:
    
    - Image_filename -> Image name (including the extension)
    - Feature__0001, Feature__0002, ... -> the feature values
    
    Parameters
    ----------
    name: str
        The name of the feature set 
    src_images: iterable of str (N)
        Relative or absolute full paths to the the source images
    case_ids: iterable of str (N)
        The case id corresponding to each image
    dst_folder: str
        Path to the folder where the csv cotaining the feature values
        will be stored. The folder is created automatically if not 
        present.
    feature_extractor: object
        The feature extractor object. It is assumed that the object
        implements the get_features() function. The function should
        take as input a cenotaph.basics.base_classes.Image and return a
        1d array of features.
    overwrite: bool
        If False the features are not computed if a name.csv file is 
        already present in the dst_folder. Otherwise the features are 
        computed anyway and the file (if present) is overwritten.
    verbose: bool
        Verbose output. If True displays the file names as they are 
        processed.

    
    Returns
    -------
    None
    """     
    
    #Create the destination folder if it doesn't exist
    if not os.path.isdir(dst_folder):
        os.makedirs(dst_folder)
    
    #Store the image name and feature value here
    df_features = pd.DataFrame()
    
    #Compute the features only if the destination file does not exist
    #or overwrite=True
    dst_file = f'{dst_folder}/{name}.csv'
    if (not os.path.isfile(dst_file)) or overwrite:
        for src_image in src_images:
            
            #Read the input image
            img = Image(src_image)
            
            #Compute the features
            feature_values = feature_extractor.get_features(img)
            
            #Get the number of features and the number of digits required
            #to number the features
            n_features = len(feature_values)
            n_digits = len(str(n_features))
            
            #Copy the feature values into the corresponding dataframe
            #columns
            feature_keys = [f'Feature__{str(i).zfill(n_digits)}' for i in 
                            range(len(feature_values))]  
            
            #Append to the dataframe
            record = {'Image_filename': os.path.basename(src_image)}
            record.update(dict(zip(feature_keys, feature_values)))
            df_features = pd.concat([df_features, 
                                     pd.DataFrame(record, index=[0])])
            
            if verbose:
                msg = f'Computing features: [{name}] on [{src_image}]'
                print(msg)
        
        #Insert the case id at the beginning in the DataFrame
        df_features.insert(loc=0, column='CaseID',
                           value=case_ids.tolist())
        
        #Save the dataframe
        df_features.to_csv(path_or_buf=f'{dst_folder}/{name}.csv', 
                           index=False)
        
    return None
        
def estimate_accuracy_k_fold(df_features, df_metadata, clf, scaler,
                             n_folds, splits_file, pattern_id='CaseID', 
                             target_label='Malignancy', 
                             feature_prefix='Feature__'):
    """Internal validation on one dataset based on by k-fold train/test 
    splitting. The folds are stratified by the values of target_label.
    
    Parameters
    ----------
    splits_file: str
        Path to where the train/test splits are stored (for repeatable
        results)
    df_features: pd.DataFrame
        The datframe containing the features.
    df_metadata: pd.DataFrame
        The dataframe containing the metadata.
    clf: object
        The classifiesr object as for instance provided by scikit-learn.
    scaler: object or None
        A scaler object as for instance provided by scikit-learn. Pass 
        None for no scaling.
    n_folds: int
        The number of folds.
    pattern_id: str
        The name of the column that uniquely identifies each pattern 
        (i.e., case, instance, etc) across train_df, test_df and 
        metadata_df.
    target_label: str
        The name of the column that encodes the class labels in the
        metadata
    feature_prefix: str
        Prefix that identifies the columns that contain feature in
        df_train and df_test
        
    Returns
    -------
    acc: 1d array-like of floats
    sensitivity: 1d array-like of floats
    specificity: 1d array-like of floats  
    """
    
    #Initialise the return values
    acc, sensitivity, specificity = list(), list(), list()
    
    #Set the primary key as index
    for df in [df_features, df_metadata]:
        df.set_index(keys=pattern_id, inplace=True)  
        
    #Sort by index to ensure repeatability
    df_features.sort_index(inplace=True)
    
    #Prepare the splits
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, 
                               random_state=0)
    X=df_features.index
    y=df_metadata.loc[X][target_label]
    
    #Check if the train/test splits exists, if not generate them
    if os.path.isfile(splits_file):
        df_splits = pd.read_csv(filepath_or_buffer=splits_file)
    else:
        df_splits = pd.DataFrame(index=df_features.index)
        for i, (train_idxs, test_idxs) in enumerate(splitter.split(X, y)):
            train_instances = df_features.index[train_idxs]
            test_instances = df_features.index[test_idxs]
            current_split_col_name = f'Split_{i}'
            df_splits[current_split_col_name] = 'unassigned'
            df_splits.loc[train_instances, current_split_col_name] = 'train' 
            df_splits.loc[test_instances, current_split_col_name] = 'test'
        
        #Create the destination folder if it doesn't exist and save the
        #splits
        splits_dst_folder = os.path.dirname(splits_file)
        if not os.path.isdir(splits_dst_folder):
            os.makedirs(splits_dst_folder)
            df_splits.to_csv(splits_file)
        
    #Read the splits and iterate through them
    splits = pd.read_csv(splits_file, index_col=pattern_id)
    for split in splits.columns:
        train_indices_on_df = df_features.index[splits[split] == 'train']
        test_indices_on_df = df_features.index[splits[split] == 'test']
        
        df_train = df_features.loc[train_indices_on_df]
        df_test = df_features.loc[test_indices_on_df]
        df_train_metadata = df_metadata.loc[train_indices_on_df]
        df_test_metadata = df_metadata.loc[test_indices_on_df]
        
        _, _acc, _sens, _spec = estimate_accuracy(
            df_train=df_train, df_test=df_test, 
            df_train_metadata=df_train_metadata, 
            df_test_metadata=df_test_metadata, clf=clf, scaler=scaler,
            pattern_id=pattern_id, target_label=target_label, 
            feature_prefix=feature_prefix)
        
        acc.append(_acc)
        sensitivity.append(_sens)
        specificity.append(_spec)
        
    #Reset the indices
    for df in [df_features, df_metadata]:
        df.reset_index(inplace=True)  
        
    return acc, sensitivity, specificity
    
            
def estimate_accuracy(df_train, df_test, df_train_metadata, 
                      df_test_metadata, clf, scaler,
                      pattern_id='CaseID', target_label='Malignancy',
                      feature_prefix='Feature__'):
    """Estimates accuracy given train/test labels and features
    
    Parameters
    ----------
    df_train: pd.DataFrame
        The datframe containing the train data.
    df_test: pd.DataFrame
        The dataframe containing the test data.
    df_train_metadata: pd.DataFrame
        The dataframe containing the metadata of the train set.
    df_test_metadata: pd.DataFrame
        The dataframe containing the metadata of the test set.
    clf: object
        The classifiesr object as for instance provided by scikit-learn
    scaler: object or None
        A scaler object as for instance provided by scikit-learn. Pass 
        None for no scaling.
    pattern_id: str
        The name of the column that uniquely identifies each pattern 
        (i.e., case, instance, etc) across train_df, test_df and 
        metadata_df.
    target_label: str
        The name of the column that encodes the class labels in the
        metadata
    feature_prefix: str
        Prefix that identifies the columns that contain feature in
        df_train and df_test
    
    Returns
    -------
    y_pred: 1d array-like
        The predicted labels
    acc: float
    sensitivity: float
    specificity: float   
    """
    #Set the primary key as index
    for df in [df_train, df_test, df_train_metadata, df_test_metadata]:
        if df.index.name != pattern_id:
            df.set_index(keys=pattern_id, inplace=True)
    
    #Columns that represent features
    feature_cols = [c for c in df_train if c.startswith(feature_prefix)]
    
    #Get the train features and labels
    X_train = df_train[feature_cols]
    y_train = df_train_metadata.loc[df_train.index][target_label]
    
    #Get the test features
    X_test = df_test[feature_cols]
    
    #Apply feature normalisation if required
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    #Train the model
    clf.fit(X=X_train, y=y_train)
    
    #Predict the labels in the test set
    y_pred = clf.predict(X=X_test)
    y_true = df_test_metadata.loc[df_test.index][target_label]
    
    #Compute the performance metrics
    tn, fp, fn, tp = cm(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    
    #Reset the indices
    for df in [df_train, df_test, df_train_metadata, df_test_metadata]:
        df.reset_index(inplace=True)    
    
    return y_pred, acc, sensitivity, specificity
        
    