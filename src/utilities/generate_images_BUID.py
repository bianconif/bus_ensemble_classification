#Generate images from the BrEaST dataset
import cv2
import os
import pandas as pd

from breast_ultrasound.src.functions import bbox_around_mask,\
     mask_bounding_box

from common import border_around_bbox

images_source_folder = (f'../../../../../../../Disk_I/LACIE/ImageDatasets/Texture/Biomedical/BUS/breast-ultrasound-image-database-BUID/images')
images_dest_folder = '../data/datasets/BUID/images'
metadata_dest_folder = '../data/datasets/BUID/metadata'
for folder in [images_dest_folder, metadata_dest_folder]:
    if not os.path.isdir(folder):
        os.makedirs(folder)


def get_case_id_and_label(s):
    sep = ' '
    all_splits = s.split(sep)
    case_id = all_splits[0] + sep + all_splits[1]
    label = all_splits[1]
    return case_id, label

df_metadata_out = pd.DataFrame()

#Get the file names
fnames = os.listdir(path=images_source_folder)

#Get the case IDs
case_ids_and_labels = {get_case_id_and_label(f) for f in fnames}

for case_id, label in case_ids_and_labels:
    
    #Read the original image
    img = cv2.imread(f'{images_source_folder}/{case_id} Image.bmp', 
                     cv2.IMREAD_GRAYSCALE)   
    
    #Read the mask
    mask = cv2.imread(f'{images_source_folder}/{case_id} Mask.tif', 
                      cv2.IMREAD_GRAYSCALE)    
    
    #Crop the original image and the mask to the mask's bounding box
    #plus a user-defined offset
    bbox = bbox_around_mask(mask=mask, offset=border_around_bbox)
    ul, lr = mask_bounding_box(bbox)
    cropped_img = img[ul[0]:lr[0],ul[1]:lr[1]]
    cropped_roi = mask[ul[0]:lr[0],ul[1]:lr[1]] * 255
    cropped_mask = 0 * cropped_roi + 255
    
    #Save the image and the mask
    img_name = f'{case_id}_img.png'
    mask_name = f'{case_id}_mask.png'
    roi_name = f'{case_id}_roi.png'
    
    cv2.imwrite(f'{images_dest_folder}/{img_name}', cropped_img)
    cv2.imwrite(f'{images_dest_folder}/{mask_name}', cropped_mask) 
    cv2.imwrite(f'{images_dest_folder}/{roi_name}', cropped_roi) 
    
    #Add record to the out dataframe
    record = {'CaseID': case_id,
              'Malignancy': label == 'Malignant',
              'Image_filename': img_name,
              'Mask_filename': mask_name,
              'Roi_filename': roi_name}
    df_metadata_out = pd.concat((df_metadata_out,
                                 pd.DataFrame(data=record, index=[0])))
    
df_metadata_out.to_csv(path_or_buf=f'{metadata_dest_folder}/metadata.csv',
                       index=False)
    
    