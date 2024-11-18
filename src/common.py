import os

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

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

#LIFEx feature table
lifex_features = {
    'intensity-based': 
    ['INTENSITY-BASED_MeanIntensity(IBSI:Q4LE)',
         'INTENSITY-BASED_IntensityVariance(IBSI:ECT3)',
         'INTENSITY-BASED_IntensitySkewness(IBSI:KE2A)',
         'INTENSITY-BASED_IntensityKurtosis(IBSI:IPH6)',
         'INTENSITY-BASED_MedianIntensity(IBSI:Y12H)',
         'INTENSITY-BASED_MinimumIntensity(IBSI:1GSF)',
         'INTENSITY-BASED_10thIntensityPercentile(IBSI:QG58)',
         'INTENSITY-BASED_25thIntensityPercentile(IBSI:No)',
         'INTENSITY-BASED_50thIntensityPercentile(IBSI:Y12H)',
         'INTENSITY-BASED_75thIntensityPercentile(IBSI:No)',
         'INTENSITY-BASED_90thIntensityPercentile(IBSI:8DWT)',
         'INTENSITY-BASED_StandardDeviation(IBSI:No)',
         'INTENSITY-BASED_MaximumIntensity(IBSI:84IY)',
         'INTENSITY-BASED_IntensityInterquartileRange(IBSI:SALO)',
         'INTENSITY-BASED_IntensityRange(IBSI:2OJQ)',
         'INTENSITY-BASED_IntensityBasedMeanAbsoluteDeviation(IBSI:4FUA)',
         'INTENSITY-BASED_IntensityBasedRobustMeanAbsoluteDeviation(IBSI:1128)',
         'INTENSITY-BASED_IntensityBasedMedianAbsoluteDeviation(IBSI:N72L)',
         'INTENSITY-BASED_IntensityBasedCoefficientOfVariation(IBSI:7TET)',
         'INTENSITY-BASED_IntensityBasedQuartileCoefficientOfDispersion(IBSI:9S40)',
         'INTENSITY-BASED_AreaUnderCurveCIVH(IBSI:No)',
         'INTENSITY-BASED_IntensityBasedEnergy(IBSI:N8CA)',
         'INTENSITY-BASED_RootMeanSquareIntensity(IBSI:5ZWQ)'],
    'intensity-histogram': 
    ['INTENSITY-HISTOGRAM_IntensityHistogramMean(IBSI:X6K6)',
         'INTENSITY-HISTOGRAM_IntensityHistogramVariance(IBSI:CH89)',
         'INTENSITY-HISTOGRAM_IntensityHistogramSkewness(IBSI:88K1)',
         'INTENSITY-HISTOGRAM_IntensityHistogramKurtosis(IBSI:C3I7)',
         'INTENSITY-HISTOGRAM_IntensityHistogramMedian(IBSI:WIFQ)',
         'INTENSITY-HISTOGRAM_IntensityHistogram10thPercentile(IBSI:GPMT)',
         'INTENSITY-HISTOGRAM_IntensityHistogram25thPercentile(IBSI:No)',
         'INTENSITY-HISTOGRAM_IntensityHistogram50thPercentile(IBSI:No)',
         'INTENSITY-HISTOGRAM_IntensityHistogram75thPercentile(IBSI:No)',
         'INTENSITY-HISTOGRAM_IntensityHistogram90thPercentile(IBSI:OZ0C)',
         'INTENSITY-HISTOGRAM_IntensityHistogramStd(IBSI:No)',
         'INTENSITY-HISTOGRAM_IntensityHistogramMode(IBSI:AMMC)',
         'INTENSITY-HISTOGRAM_IntensityHistogramInterquartileRange(IBSI:WR0O)',
         'INTENSITY-HISTOGRAM_IntensityHistogramMeanAbsoluteDeviation(IBSI:D2ZX)',
         'INTENSITY-HISTOGRAM_IntensityHistogramRobustMeanAbsoluteDeviation(IBSI:WRZB)',
         'INTENSITY-HISTOGRAM_IntensityHistogramMedianAbsoluteDeviation(IBSI:4RNL)',
         'INTENSITY-HISTOGRAM_IntensityHistogramCoefficientOfVariation(IBSI:CWYJ)',
         'INTENSITY-HISTOGRAM_IntensityHistogramQuartileCoefficientOfDispersion(IBSI:SLWD)',
         'INTENSITY-HISTOGRAM_IntensityHistogramEntropyLog10(IBSI:No)',
         'INTENSITY-HISTOGRAM_IntensityHistogramEntropyLog2(IBSI:TLU2)',
         'INTENSITY-HISTOGRAM_AreaUnderCurveCIVH(IBSI:No)',
         'INTENSITY-HISTOGRAM_Uniformity(IBSI:BJ5W)',
         'INTENSITY-HISTOGRAM_RootMeanSquare(IBSI:No)',
         'INTENSITY-HISTOGRAM_MaximumHistogramGradient(IBSI:12CE)',
         'INTENSITY-HISTOGRAM_MaximumHistogramGradientGreyLevel(IBSI:8E6O)',
         'INTENSITY-HISTOGRAM_MinimumHistogramGradient(IBSI:VQB3)',
         'HISTOGRAM_MinimumHistogramGradientGreyLevel(IBSI:RHQZ)'],
    'GLCM': 
    ['GLCM_JointMaximum(IBSI:GYBY)', 
         'GLCM_JointAverage(IBSI:60VM)',
         'GLCM_JointVariance(IBSI:UR99)',
         'GLCM_JointEntropyLog2(IBSI:TU9B)',
         'GLCM_JointEntropyLog10(IBSI:No)',
         'GLCM_DifferenceAverage(IBSI:TF7R)',
         'GLCM_DifferenceVariance(IBSI:D3YU)',
         'GLCM_DifferenceEntropy(IBSI:NTRS)',
         'GLCM_SumAverage(IBSI:ZGXS)',
         'GLCM_SumVariance(IBSI:OEEB)',
         'GLCM_SumEntropy(IBSI:P6QZ)',
         'GLCM_AngularSecondMoment(IBSI:8ZQL)',
         'GLCM_Contrast(IBSI:ACUI)',
         'GLCM_Dissimilarity(IBSI:8S9J)',
         'GLCM_InverseDifference(IBSI:IB1Z)',
         'GLCM_NormalisedInverseDifference(IBSI:NDRX)',
         'GLCM_InverseDifferenceMoment(IBSI:WF0Z)',
         'GLCM_NormalisedInverseDifferenceMoment(IBSI:1QCO)',
         'GLCM_InverseVariance(IBSI:E8JP)',
         'GLCM_Correlation(IBSI:NI2N)',
         'GLCM_Autocorrelation(IBSI:QWB0)',
         'GLCM_ClusterTendency(IBSI:DG8W)',
         'GLCM_ClusterShade(IBSI:7NFM)',
         'GLCM_ClusterProminence(IBSI:AE86)'],
    'GLRLM': 
    ['GLRLM_ShortRunsEmphasis(IBSI:22OV)',
         'GLRLM_LongRunsEmphasis(IBSI:W4KF)',
         'GLRLM_LowGreyLevelRunEmphasis(IBSI:V3SW)',
         'GLRLM_HighGreyLevelRunEmphasis(IBSI:G3QZ)',
         'GLRLM_ShortRunLowGreyLevelEmphasis(IBSI:HTZT)',
         'GLRLM_ShortRunHighGreyLevelEmphasis(IBSI:GD3A)',
         'GLRLM_LongRunLowGreyLevelEmphasis(IBSI:IVPO)',
         'GLRLM_LongRunHighGreyLevelEmphasis(IBSI:3KUM)',
         'GLRLM_GreyLevelNonUniformity(IBSI:R5YN)',
         'GLRLM_RunLengthNonUniformity(IBSI:W92Y)',
         'GLRLM_RunPercentage(IBSI:9ZK5)'],
    'NGTDM': 
    ['NGTDM_Coarseness(IBSI:QCDE)',
         'NGTDM_Contrast(IBSI:65HE)',
         'NGTDM_Busyness(IBSI:NQ30)',
         'NGTDM_Complexity(IBSI:HDEZ)',
         'NGTDM_Strength(IBSI:1X9X)'],
    'GLSZM': 
    ['GLSZM_SmallZoneEmphasis(IBSI:5QRC)',
         'GLSZM_LargeZoneEmphasis(IBSI:48P8)',
         'GLSZM_LowGrayLevelZoneEmphasis(IBSI:XMSY)',
         'GLSZM_HighGrayLevelZoneEmphasis(IBSI:5GN9)',
         'GLSZM_SmallZoneLowGreyLevelEmphasis(IBSI:5RAI)',
         'GLSZM_SmallZoneHighGreyLevelEmphasis(IBSI:HW1V)',
         'GLSZM_LargeZoneLowGreyLevelEmphasis(IBSI:YH51)',
         'GLSZM_LargeZoneHighGreyLevelEmphasis(IBSI:J17V)',
         'GLSZM_GreyLevelNonUniformity(IBSI:JNSA)',
         'GLSZM_NormalisedGreyLevelNonUniformity(IBSI:Y1RO)',
         'GLSZM_ZoneSizeNonUniformity(IBSI:4JP3)',
         'GLSZM_NormalisedZoneSizeNonUniformity(IBSI:VB3A)',
         'GLSZM_ZonePercentage(IBSI:P30P)',
         'GLSZM_GreyLevelVariance(IBSI:BYLV)',
         'GLSZM_ZoneSizeVariance(IBSI:3NSA)',
         'GLSZM_ZoneSizeEntropy(IBSI:GU8N)'],
}	



single_descriptors = [*morphological_features.keys(), 
                      *texture_descriptors.keys(),
                      *cnn_descriptors.keys(),
                      *lifex_features.keys()]

#========================================
#========================================
#========================================

#========================================
#===== Combination of descriptors) ======
#========================================
combined_descriptors = {
    #'ConvNeXt_base+Morphological+HOG': ['ConvNeXt_base', 'Morphological', 
                                        #'HOG'],
    'ConvNeXt_base+Morphological+HOG+IH': [
        'ConvNeXt_base', 'Morphological', 'HOG', 'intensity-histogram']    
}

#fusion_methods = ['early-fusion', 'majority-voting', 'prod', 'sum']
fusion_methods = ['early-fusion']
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
    ),
    #'Logistic regression':
    #ClassifierWrapper(
        #classifier = LogisticRegression(),
        #param_grid = {'C': [0.1, 1.0, 10.0, 100.0]}
    #)     
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

#===========================================================
#============= LIFEx feature settings ======================
#===========================================================

#Subfolder where the raw LIFEx features are stored
lifex_raw_features_subfolder = 'lifex_raw_features'

#File containing the raw LIFEx features
lifex_raw_feature_file = 'lifex_raw_features.csv'

