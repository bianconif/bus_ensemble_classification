#Generate images from the BrEaST dataset
import cv2
import os
import pandas as pd

from breast_ultrasound.src.functions import bbox_around_mask,\
     mask_bounding_box

from common import border_around_bbox, patch_distance, patch_size
from functions import denoise_nlm

images_source_folder = (f'../../breast_ultrasound/data/images/'
                        f'pre-processed/resized')
images_dest_folder = '../data/datasets/BrEaST/images'
metadata_dest_folder = '../data/datasets/BrEaST/metadata'
for folder in [images_dest_folder, metadata_dest_folder]:
    if not os.path.isdir(folder):
        os.makedirs(folder)


metadata_src = (f'../../breast_ultrasound/data/metadata/'
                f'BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx')

#Read the metadata
df_metadata_in = pd.read_excel(io=metadata_src,
                               sheet_name='BrEaST-Lesions-USG clinical dat')

#Drop the normal cases (no mask)
df_metadata_in.dropna(subset="Mask_tumor_filename", inplace=True)

df_metadata_out = pd.DataFrame()

for _, row in df_metadata_in.iterrows():
    
    #Read the original image
    img = cv2.imread(f'{images_source_folder}/{row["Image_filename"]}', 
                     cv2.IMREAD_GRAYSCALE)   
    
    #Read the mask
    mask = cv2.imread(
        f'{images_source_folder}/{row["Mask_tumor_filename"]}', 
        cv2.IMREAD_GRAYSCALE)    
    
    #Crop the original image and the mask to the mask's bounding box
    #plus a user-defined offset
    bbox = bbox_around_mask(mask=mask, offset=border_around_bbox)
    ul, lr = mask_bounding_box(bbox)
    cropped_img = img[ul[0]:lr[0],ul[1]:lr[1]]
    cropped_roi = mask[ul[0]:lr[0],ul[1]:lr[1]] * 255
    cropped_mask = 0 * cropped_roi + 255
    
    #Apply denoising
    cropped_img = denoise_nlm(img_in=cropped_img, patch_size=patch_size, 
                              patch_distance=patch_distance)
    
    #Save the image and the mask
    img_name = f'{row["CaseID"]:03d}_img.png'
    mask_name = f'{row["CaseID"]:03d}_mask.png'
    roi_name = f'{row["CaseID"]:03d}_roi.png'
    
    cv2.imwrite(f'{images_dest_folder}/{img_name}', cropped_img)
    cv2.imwrite(f'{images_dest_folder}/{mask_name}', cropped_mask) 
    cv2.imwrite(f'{images_dest_folder}/{roi_name}', cropped_roi) 
    
    #Add record to the out dataframe
    record = {'CaseID': row["CaseID"],
              'Malignancy': row['Classification'] == 'malignant',
              'Image_filename': img_name,
              'Mask_filename': mask_name,
              'Roi_filename': roi_name}
    df_metadata_out = pd.concat((df_metadata_out,
                                 pd.DataFrame(data=record, index=[0])))
    
df_metadata_out.to_csv(path_or_buf=f'{metadata_dest_folder}/metadata.csv',
                       index=False)
    
    