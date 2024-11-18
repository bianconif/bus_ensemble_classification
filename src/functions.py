import numpy as np
import os
import pandas as pd
from skimage.restoration import denoise_nl_means, estimate_sigma
from statsmodels.stats.proportion import proportion_confint

from PIL import Image

def split_image(img, num_splits):
    """Splits an image into non-overlapping tiles (partition)
    
    Parameters
    ----------
    img : int (H,W)
        The input image
    num_splits : int (2)
        The number of splits in height and width respectively
        
    Returns
    -------
    tiles : list of length num_splits[0] * num_splits[1]
        The image tiles
    """
    
    #Compute the size of the central crop
    h, w = [i - i % j for i,j in zip(img.shape, num_splits)]
    
    #Compute the size of the tiles
    tile_height, tile_width = h//num_splits[0], w//num_splits[1]
    
    #Centre-crop the original image
    H, W = img.shape[0:2] 
    ul = (np.floor((H-h)/2).astype(np.uint),
          np.floor((W-w)/2).astype(np.uint))
    cropped_img = img[ul[0]:ul[0] + h, ul[1]:ul[1] + w]
    
    #Get the tiles
    tiles = list()
    for i in range(0, cropped_img.shape[0], tile_height):  
        for j in range(0, cropped_img.shape[1], tile_width):
            tiles.append(cropped_img[i:i+tile_height, 
                                     j:j+tile_width])       
    return tiles

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

def compute_features(name, src_images, pattern_ids, src_image_column,
                     pattern_id_column, dst_folder, 
                     feature_extractor, feature_prefix, 
                     overwrite=False, verbose=True):
    """Computes features. Generates a name.csv file in the dst_folder 
    where each row corresponds to one image (pattern). Columns are 
    organised as follows:
    
    - `pattern_id_column` -> The pattern ids
    - `src_image_column` -> The source image
    - `{feature_prefix}0`, `{feature_prefix}1`, ... 
      `{feature_prefix}n-1`-> the feature values
    
    Parameters
    ----------
    name: str
        The name of the feature set 
    src_images: iterable of str (N)
        Relative or absolute full paths to the the source images.
    pattern_ids: iterable of str (N)
        Labels that uniquely identifies each pattern (image).
    src_image_column: str
        Name of the column that stores the image filename in the 
        generated file.
    pattern_id_column: str
        Name of the column that stores the pattern id in the 
        generated file.
    dst_folder: str
        Path to the folder where the csv cotaining the feature values
        will be stored. The folder is created automatically if not 
        present.
    feature_extractor: object
        The feature extractor object. It is assumed that the object
        implements the get_features() function. The function should
        take as input a PIL.Image and return a
        1d array of features.
    feature_prefix: str
        Prefix to identify the feature columns in the generated dataframe.
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
            img = Image.open(src_image)
            
            #Compute the features
            feature_values = feature_extractor.get_features(img)
            
            #Get the number of features and the number of digits required
            #to number the features
            n_features = len(feature_values)
            n_digits = len(str(n_features))
            
            #Copy the feature values into the corresponding dataframe
            #columns
            feature_keys = [f'{feature_prefix}{str(i).zfill(n_digits)}' 
                            for i in range(len(feature_values))]  
            
            #Append to the dataframe
            record = dict()
            record.update(dict(zip(feature_keys, feature_values)))
            df_features = pd.concat([df_features, 
                                     pd.DataFrame(record, index=[0])])
            
            if verbose:
                msg = f'Computing features: [{name}] on [{src_image}]'
                print(msg)
        
        #Insert the case id at the beginning in the DataFrame
        df_features.insert(loc=0, column=pattern_id_column,
                           value=pattern_ids.tolist())
        
        #Save the dataframe
        df_features.to_csv(path_or_buf=f'{dst_folder}/{name}.csv', 
                           index=False)
        
    return None
        
def get_feature_columns(df_features, feature_prefix):
    """Extracts the feature columns from a feature dataframe
    
    Parameters
    ----------
    df_features: pd.DataFrame
        The feature dataframe.
    feature_prefix: src
        The prefix that identifies the columns containing the features
        in the feature dataframe.
        
    Returns
    -------
    feature_columns: list of str
        List of the columns that contain the features sorted in 
        alphabetical order.  
    """    
    feature_columns = [c for c in df_features.columns if 
                       c.startswith(feature_prefix)]
    return sorted(feature_columns)

def separable_filters(singledim):
    """Combine a sequence of one-dimensional filters into a bank of 
    two-dimensional ones.

    Parameters
    ----------
    singledim : ndarray (sz, N) 
        One dimensional filters of size N

    Returns
    -------
    ndarray (N, N, M**2) 
        The two-dimensional filters
    """
    sz = singledim.shape[1]
    f = np.einsum('ai, bj -> ijab', singledim, singledim)
    return f.reshape([sz, sz, -1])

def pack_results(acc, sens, spec, n_test_samples, alpha, ci_method):
    """Generates a dict containing the summary performance metrics.
    
    Parameters
    ----------
    acc, sens, spec: float
        Accuracy, sensitivity and specificity values defined in [0,1].
    valid_res: nparray (n_splits,3)
        The array of results for each split, where columns represent
        accuracy, sensitivity and specificity respectively.
    n_test_samples: int
        Number of samples tested (for estimating confidence interval
        for accuracy).
    alpha: float
        Significance level for accuracy's confidence intervals.
    ci_method: str
        The method used for estimating accuracy's confidence interval. See
        proportion_confint for possible values.
        
    Returns
    -------
    results_dict: dict
        Dictionary summarising the results.
    """
    
    #Estimate confidence interval fro accuracy
    acc_ci = proportion_confint(count=int(acc*n_test_samples), 
                                nobs=n_test_samples,
                                alpha=alpha,
                                method=ci_method)    
    
    #Convert to percentages
    acc, sens, spec, *acc_ci = 100*np.array((acc, sens, spec, *acc_ci))
    
    results_dict = {'Acc.': acc, 
                    'Acc. CI_l': acc_ci[0], 
                    'Acc. CI_u': acc_ci[1],
                    'Sens.': sens, 
                    'Spec.': spec}
    
    return results_dict
    
def retrieve_lifex_features(feature_names, lifex_out_file, 
                            feature_prefix, pattern_id_column,
                            parser):
    """Retrieves selected features from a LIFEx-generated output file
    
    Parameters
    ----------
    feature_names : list of str
        The names of the features to retrieve.
    lifex_out_file : str
        Relative or absolute path to the output file generated by
        LIFEx (v 7.6.0)
    feature_prefix: str
        Prefix to identify the feature columns in the generated dataframe.
    pattern_id_column: str
        Name of the column that identifies the unique pattern id in the
        generated dataframe.
    parser: object
        The dataset-specific function that parses the `INFO_NameOfRoi`
        column in the LIFEX-generated file
        
    Returns
    -------
    df_features : pandas.DataFrame
        The retrieved features.
    """
    df = pd.read_csv(filepath_or_buffer=lifex_out_file)
    to_keep = [pattern_id_column]
    
    #Add the pattern column
    df[pattern_id_column] = df['INFO_NameOfRoi'].apply(parser)
    
    for feature_name in feature_names:
        for column in df.columns:
            if feature_name in column:
                to_keep.append(column)    
                break;
            
    df_features = df[to_keep]
    
    #Prepend feature prefix to the feature names
    mapper = {feature: f'{feature_prefix}{feature}' for feature in
              to_keep if feature != pattern_id_column}
    df_features.rename(columns=mapper, inplace=True)   
        
    return df_features  
    
    
    