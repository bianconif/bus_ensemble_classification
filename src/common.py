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