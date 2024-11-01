import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma

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