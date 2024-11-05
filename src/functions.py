import numpy as np
import os
import pandas as pd
from skimage.restoration import denoise_nl_means, estimate_sigma

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
        
        
    