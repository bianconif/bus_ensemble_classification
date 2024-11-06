from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cenotaph.texture.filtering import DCF, Gabor, Laws

from classes import DescriptorWrapper

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

#Column that identifies the class label in the the feature and metadata 
#files
class_column = 'Malignancy'

#Class labels for positive (malignant) and negative (benign) 
#classes respectively
binary_class_labels = ('1','0')

#========================================
#=========== Feature sets ===============
#========================================
texture_descriptors = {
    'Gabor':
    DescriptorWrapper(
        descriptor=Gabor(size=11, scales = 6, complex_=True),
        mode='image'
    )
}          

descriptors = texture_descriptors
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
    {'train': datasets[0], 'test': datasets[0]}
]

scalers = {
    'z-score': StandardScaler()
}

clfs = {
    'rbfSVM': SVC()
}
#========================================
#========================================
#========================================