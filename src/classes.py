from cv2 import getGaborKernel
from dataclasses import dataclass
import numpy as np

from scipy.ndimage import convolve
from skimage.measure import regionprops_table
from skimage.feature import hog
from skimage.feature import local_binary_pattern


from PIL import Image, ImageOps
from torchvision.models.feature_extraction import create_feature_extractor
import torch

from functions import split_image


@dataclass
class DescriptorWrapper:
    """Wrapper around an image descriptor.
    
    Attributes
    ----------
    descriptor: object
        The descriptor object -- i.e., the object that computes the 
        features
    mode: str
        Whether the features should be computed from the image or the roi.
        Can be either 'image' or 'roi'
    """
    descriptor: object
    mode: str

@dataclass    
class ClassifierWrapper:
    """Wrapper around a classifier.
    
    Attributes
    ----------
    classifier: object
        The classifier object. An untrained instance of a classifier
        object as provided by sckiti-learn, or any other object which 
        implements that interface.
    param_grid: dict or None
        Dictionary with parameters names (str) as keys and lists of 
        parameter settings to try as values. Used for hyperparameter
        tuning by exhaustive search. Choose None if tuning is not 
        required.
    """
    classifier: object
    param_grid: dict
       
class Gabor:
    """Computes features based on Gabor filters"""
    
    def __init__(self, n_freqs, min_freq, max_freq, 
                 n_ornts, ksize=11, sigma=2.0, gamma=1.0):
        """
        Parameters
        ----------
        n_freqs: int
            The number of frequencies.
        min_freq: float
            The minimum frequency of the filter bank in px**-1.
        max_freq: float
            The maximum frequency of the filter bank in px**-1. 
        n_ornts: int
            The number of orientations
        ksize: tuple of int (W,H)
            The size of the filter kernel (width, height) in pixels. 
        sigma: float
            The standard deviation of the Gaussian envelope.
        gamma: float
            The spatial aspect ratio (ellipticity) of the filter.
        """
        
        #Compute the frequency and orientation of each filter from the
        #given parameters
        freqs = np.linspace(start=min_freq, stop=max_freq, 
                            num=n_freqs, endpoint=True)    
        ornts = np.linspace(start=0.0, stop=np.pi/2, num=n_ornts, 
                            endpoint=False)            
        
        #Prepare the filter bank's kernels. Observe that OpenCV returns 
        #the real part of the Gabor filter
        self._kernels = list()
        for ornt in ornts:
            for freq in freqs:
                kernel = getGaborKernel(
                    ksize=ksize, lambd=1/freq, theta=ornt, 
                    sigma=sigma, gamma=gamma)
                self._kernels.append(kernel)
       
    def get_features(self, img):
        """
        Parameters
        ----------
        img: PIL.Image
            The input image.
        
        Returns
        -------
        features: iterable of float
            The features
        """         
        
        features = list()
        img = np.asarray(ImageOps.grayscale(img))       
        
        #Iterate through the filter bank and compute the transformed images
        for kernel in self._kernels:
                    
            #Compute the transformed image
            transformed_image = convolve(img, weights=kernel, 
                                         mode='wrap')
                    
            features.append(transformed_image.mean())
            features.append(transformed_image.std())  
            
        return features
        
    
class HOG:
    """Computes features based on histograms of oriented gradients"""
    
    def __init__(self, num_splits=(3,3), num_ornts=9):
        """
        Parameters
        ----------
        num_splits : tuple of ints (2)
            The number of vertical and horizontal subdivisions into which
            the original image is split. Histograms of oriented gradients
            are computed from each tile and the resulting features 
            concatenated.
        num_ornts : int
            The number of orientations over which the HOG are computed.
        """
        self._num_splits = num_splits
        self._num_ornts = num_ornts
        

    def get_features(self, img):
        """
        Parameters
        ----------
        img: PIL.Image
            The input image.
        
        Returns
        -------
        features: iterable of float
            The features
        """         
        
        features = list()
        img = np.asarray(ImageOps.grayscale(img))
        
        for tile in split_image(img=img, 
                                num_splits=self._num_splits):
            hog_response = hog(
                tile,
                pixels_per_cell=(tile.shape[0], tile.shape[1]),
                cells_per_block=(1,1),
                block_norm='L1',
                feature_vector=True)
        
            features.extend(hog_response)  
            
        return features

class LBP:
    """Computes features based on histograms of equivalent patterns"""
    
    def __init__(self, n_points=8, radius=1, method='ror'):
        """
        Parameters
        ----------
        n_points : int
            Number of points in the circular neighbourhood.
        radius : int
            The radius of the circular neighbourhood (in px).
        method : str
            See skimage.feature.local_binary_pattern() for the meaning and
            possible values.
        """
        self._n_points = n_points
        self._radius = radius
        self._method = method
        
        #Determine the number of possible patterns
        match (self._n_points, self._method):
            case (8, 'ror'):
                self._n_patterns = 36
            case (8, 'uniform'):
                self._n_patterns = 10
            case (8, 'nri_uniform'):
                self._n_patterns = 58        
            case _:
                raise Exception('Combination n_points/method unsupported')   
            
    def get_features(self, img):
        """
        Parameters
        ----------
        img: PIL.Image
            The input image.
        
        Returns
        -------
        features: iterable of float
        """
        
        img = np.asarray(ImageOps.grayscale(img))
        
        #Compute the LBP codes
        lbp_image = local_binary_pattern(
            image=img, P=self._n_points, R=self._radius, 
            method=self._method
        )
    
        #Flatten the codes
        lbp_codes = lbp_image.flatten()
    
        #Compute the LBP histogram
        features, _ = np.histogram(lbp_codes, density=True, 
                                   bins=self._n_patterns, 
                                   range=(0, self._n_patterns))        
        
        return features
    
class Morphological:
    """Computes morphological features from the ROI"""
    
    def __init__(self, properties):
        """
        Parameters
        ----------
        properties: iterable of str
            The morphological features to compute. For possoble values 
            see skimage.measure.regionprops_table.
        """
        self._properties = properties
        
    def get_features(self, bw_img):
        """
        Parameters
        ----------
        bw_img: cenotaph.basics.base_classes.Image
            The input image. Needs to be a two-level array where 0 
            reprents the backround, and the other value the foreground (
            i.e., the ROI)
            
        Returns
        -------
        features: iterable of float
            The features
        """
        mask = bw_img.get_data()
        props = regionprops_table(mask, 
                                  properties=('area',
                                              'area_bbox',
                                              'area_convex',
                                              'axis_major_length',
                                              'axis_minor_length',
                                              'eccentricity',
                                              'feret_diameter_max',
                                              'perimeter',
                                              'solidity'
                                              )
                                  )    
        features = [prop[0] for prop in props.values()]
        return features
    
class PreTrainedCNN:
    """Computes morphological features from pre-trained CNN"""
    
    def __init__(self, model: object, weights: object,
                 layer: str):
        self.weights = weights
        self.layer = layer
        self.model = model(weights=self.weights)

    def get_features(self, img):
        """
        Parameters
        ----------
        img: cenotaph.basics.base_classes.Image
            The grey-scale input image.
        
        Returns
        -------
        features: iterable of float
            The features
        """        

        pil_img = Image.fromarray(img.get_data())

        features = self._extract_intermediate_features(
            model=self.model, 
            weights=self.weights,
            target_layer=self.layer,
            img=pil_img
        )
    
        return features

    @staticmethod
    def _extract_intermediate_features(model, weights, target_layer, img, 
                                       norm_order=1, 
                                       resizing_method=Image.BICUBIC):
        """Extract intermediate features from a PyTorch pre-trained CNN model

        Parameters
        ----------
        model: torchvision.models
            The pre-trained PyTorch model
        weights: enum.EnumMeta
            The model's weigths     
        target_layer: str
            Name of the layer where the features are extracted from
        img: PIL Image
            The input image
        norm_order: int or None
            The order of the norm used for normalising the output. Use None
            for no normalisation.
        resizing_method: object
            The method used for resizing non-square images to the input size 
            (fov) of the CNN. See PIL.Image.resize() for possible values.

        Returns
        -------
        features: nparray of floats
            The intermediate features
        """

        #Set the model in evaluation mode
        model.eval()

        #Convert the image to three-channel
        img = np.expand_dims(a=img, axis=2)
        img = np.tile(A=img, reps=(1,1,3))
        img = PilImage.fromarray(img)

        #======= Workarund to get the input size (fov) of the CNN ===
        #Create a dummy image as a copy of the input image
        dummy_img = np.array(img).copy()
        dummy_img = PilImage.fromarray(dummy_img)

        #Preprocess the dummy image based on the model's weights
        preprocess = weights.transforms()
        dummy_input_tensor = preprocess(dummy_img)    

        #Get the input size of the CNN
        cnn_fov = list(dummy_input_tensor[0].shape)
        #============================================================

        #Resize the image to the CNN input size
        resized_img = img.resize(size=cnn_fov, resample=resizing_method)

        #Preprocess the image based on the model's weights
        input_tensor = preprocess(resized_img)

        #Create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)

        #Create the feature extractor
        return_nodes = {target_layer: 'target-layer'}
        feature_extractor = create_feature_extractor(
            model, return_nodes=return_nodes)    

        #Get the features
        intermediate_outputs = feature_extractor(input_batch)
        features = torch.flatten(intermediate_outputs['target-layer'])
        features = features.detach().numpy()

        #Apply normalisation if required
        if norm_order:
            normalisation_factor = np.linalg.norm(x=features, 
                                                  ord=norm_order)
            features = features/normalisation_factor

        return features
