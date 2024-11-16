import os

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from torchvision.models import convnext_base, ConvNeXt_Base_Weights,\
     efficientnet_v2_s, EfficientNet_V2_S_Weights, mobilenet_v2,\
     MobileNet_V2_Weights, resnet50, ResNet50_Weights,\
     swin_v2_s, Swin_V2_S_Weights

from classes import ClassifierWrapper, DCF, DescriptorWrapper, FDTA,\
     Gabor, HOG, LBP, Morphological, PreTrainedCNN

#Extent of the border around the bounding box
border_around_bbox = 4

#========================================
#==== Parameters for image denoising ====
#========================================
patch_size = 5
patch_distance = 6
#========================================
#========================================
#========================================

#Features root folder. Save the features here; there will be one subfolder
#for each dataset
features_root_folder = 'features'

#Prefix that identifies the feature columns in the feature files
feature_prefix = 'Feature__'

#Column that identifies the case id in the feature and metadata files
pattern_id_column = 'CaseID' 

#Column that identifies the image filename id in the feature files
src_image_column = 'Image_filename'

#Column that identifies the class label in the the feature and metadata 
#files
class_column = 'Malignancy'

#Class labels for positive (malignant) and negative (benign) 
#classes respectively
binary_class_labels = ('1','0')

#========================================
#===== Feature sets (descriptors) =======
#========================================

#Cache folder for LBP-like descriptors
#hep_luts = '../cache/hep_luts'
#if not os.path.isdir(hep_luts):
    #os.makedirs(hep_luts)
    

morphological_features = {
    'Morphological':
    DescriptorWrapper(
        descriptor=Morphological(
            properties=('area', 'area_bbox', 'area_convex', 
                        'axis_major_length', 'axis_minor_length',
                        'eccentricity', 'feret_diameter_max',
                        'perimeter', 'solidity')
            ),
            mode='roi'
    )
}

texture_descriptors = {
    'DCF':
    DescriptorWrapper(
        descriptor=DCF(n_freqs=5, ksize=10),
        mode='image'
    ),   
    'FDTA':
    DescriptorWrapper(
        descriptor=FDTA(num_splits=(3,3), max_res=4),
        mode='image'
    ),     
    'Gabor':
    DescriptorWrapper(
        descriptor=Gabor(n_freqs=5, min_freq=0.05, max_freq=0.35, 
                         n_ornts=8, ksize=(10,10), sigma=2.0, gamma=1.0),
        mode='image'
    ),
    'HOG':
    DescriptorWrapper(
        descriptor=HOG(),
        mode='image'
    ),    
    'LBP-8-1-nri-uniform':
    DescriptorWrapper(
        descriptor=LBP(n_points=8, radius=1, method='nri_uniform'),
        mode='image'
    ),
    'LBP-8-1-ror':
    DescriptorWrapper(
        descriptor=LBP(n_points=8, radius=1, method='ror'),
        mode='image'
    ),
    'LBP-8-1-uniform':
    DescriptorWrapper(
        descriptor=LBP(n_points=8, radius=1, method='uniform'),
        mode='image'
    ),    
    'LBP-8-2-nri-uniform':
    DescriptorWrapper(
        descriptor=LBP(n_points=8, radius=2, method='nri_uniform'),
        mode='image'
    ),
    'LBP-8-2-ror':
    DescriptorWrapper(
        descriptor=LBP(n_points=8, radius=2, method='ror'),
        mode='image'
    ),
    'LBP-8-2-uniform':
    DescriptorWrapper(
        descriptor=LBP(n_points=8, radius=2, method='uniform'),
        mode='image'
    ),     
}          

cnn_descriptors = {
    'ConvNeXt_base': 
    DescriptorWrapper(
        descriptor=PreTrainedCNN(model=convnext_base, 
                                 weights=ConvNeXt_Base_Weights.DEFAULT, 
                                 layer='avgpool'),
        mode='image'
    ),
    'EfficientNet_V2_s': 
    DescriptorWrapper(
        descriptor=PreTrainedCNN(model=efficientnet_v2_s, 
                                 weights=EfficientNet_V2_S_Weights.DEFAULT, 
                                 layer='avgpool'),
        mode='image'
    ),
    'MobileNet_V2': 
    DescriptorWrapper(
        descriptor=PreTrainedCNN(model=mobilenet_v2, 
                                 weights=MobileNet_V2_Weights.DEFAULT, 
                                 layer='adaptive_avg_pool2d'),
        mode='image'
    ),
    'ResNet50': 
    DescriptorWrapper(
        descriptor=PreTrainedCNN(model=resnet50, 
                                 weights=ResNet50_Weights.DEFAULT, 
                                 layer='avgpool'),
        mode='image'
    ),
    'Swin_V2_s': 
    DescriptorWrapper(
        descriptor=PreTrainedCNN(model=swin_v2_s, 
                                 weights=Swin_V2_S_Weights.DEFAULT, 
                                 layer='avgpool'),
        mode='image'
    )     
}


single_descriptors = {**morphological_features, 
                      **texture_descriptors,
                      **cnn_descriptors}


#========================================
#========================================
#========================================

#========================================
#===== Combination of descriptors) ======
#========================================
combined_descriptors = {
    'ConvNeXt_base+Morphological+HOG': ['ConvNeXt_base', 'Morphological', 
                                        'HOG']
}

fusion_methods = ['early-fusion', 'majority-voting', 'prod', 'sum']
#========================================
#========================================
#========================================

#========================================
#============= Datasets =================
#========================================
datasets = ['BrEaST', 'BUID']
datasets_root = '../data/datasets'
datasets_image_subfolder = 'images'
datasets_metadata_file = '/metadata/metadata.csv'
#========================================
#========================================
#========================================

#Accuracy estimation
train_test_split_method = 'stratified-k-fold'
n_splits = 5

#========================================
#======  Experimental conditions ========
#========================================
testing_conditions = [
    {'train': datasets[0], 'test': datasets[0]},
    {'train': datasets[0], 'test': datasets[1]},
    {'train': datasets[1], 'test': datasets[1]},
    {'train': datasets[1], 'test': datasets[0]}
]

scalers = {
    'z-score': StandardScaler()
}

clfs = {
    'Rbf SVC':
    ClassifierWrapper(
        classifier = SVC(kernel='rbf', gamma='auto', probability=True),
        param_grid = {'C': [0.1, 1.0, 10.0, 100.0]}
    )    
}
#========================================
#========================================
#========================================

#===========================================================
#=== Estimation of the confidence intervals for accuracy ===
#===========================================================

#Significance level for the confidence interval
acc_ci_alpha = 0.05

#Method to estimate the confidence interval
acc_ci_method = 'agresti_coull'

#===========================================================
#===========================================================
#===========================================================