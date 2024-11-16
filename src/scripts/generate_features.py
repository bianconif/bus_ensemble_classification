import pandas as pd

from common import datasets, datasets_image_subfolder,\
     datasets_metadata_file, datasets_root,\
     features_root_folder, feature_prefix, pattern_id_column
from common import cnn_descriptors, morphological_features,\
     texture_descriptors, src_image_column
from functions import compute_features

#This script computes the following classes of features. LIFEx-generated
#features are not computed here
descriptors_to_compute = {**morphological_features, 
                          **texture_descriptors,
                          **cnn_descriptors}

#Table of the combinations features/datasets
df_features_to_generate = pd.DataFrame()
for dataset in datasets:
    for descriptor_name in descriptors_to_compute:
        record = {
            'Dataset': dataset,
            'Descriptor_name': descriptor_name,
            'Image_src_folder': (f'{datasets_root}/{dataset}/'
                                 f'{datasets_image_subfolder}'),
            'Metadata_file': (f'{datasets_root}/{dataset}/'
                              f'{datasets_metadata_file}')
        }
        df_features_to_generate = pd.concat(
            (df_features_to_generate, pd.DataFrame(data=record, index=[0]))
        )
        
#Compute the features
for _, row in df_features_to_generate.iterrows():
    
    #Read the metadata
    df_metadata = pd.read_csv(row['Metadata_file'])
    
    #Get the descriptor wrapper
    descriptor_name = row['Descriptor_name']
    descriptor_wrapper = descriptors_to_compute[descriptor_name]
    
    #Get the list of input files
    match descriptor_wrapper.mode:
        case 'image':
            src_files = df_metadata['Image_filename']
        case 'roi':
            src_files = df_metadata['Roi_filename']
        case _:
            raise Exception('Mode should be either image or roi')
        
    #Prepend the path to the input files
    src_files = [row['Image_src_folder'] + '/' + src_file for src_file in 
                 src_files]
    
    #Save the features here
    dst_folder = f'{features_root_folder}/{row["Dataset"]}'
        
    compute_features(name=descriptor_name, src_images=src_files, 
                     pattern_ids=df_metadata[pattern_id_column], 
                     pattern_id_column=pattern_id_column,
                     src_image_column=src_image_column,
                     dst_folder=dst_folder, 
                     feature_extractor=descriptor_wrapper.descriptor,
                     feature_prefix=feature_prefix) 