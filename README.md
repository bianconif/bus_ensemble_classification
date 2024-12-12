# Combining conventional and CNN-based features for discriminating benign vs. malignant lesions on breast ultrasound images: An evaluation of ensemble schemes

### by [Muhammah U. Khan](https://www.linkedin.com/in/usama-khan-0a509211a/), [Francesco Bianconi](www.bianconif.net), [Hongbo Du](https://www.buckingham.ac.uk/directory/mr-hongbo-du/) and [Sabah Jassim](https://www.buckingham.ac.uk/directory/professor-sabah-jassim/)

### Submitted to...

## Companion website

## Usage

### Settings
Set the `output` folder as the initial (starting) directory in your project. Note that the main parameters that control the execution (such as input and output files/folders, list of descriptors and other settings) are stored in the `src/common.py` source file. Check and modify them if needed (but there should be no need to do so).  

### Image preparation (optional)
Use the `src/utilities/generate_images_BrEaST.py` and `src/utilities/generate_images_BUID.py` scripts to generate the images for the analysis from the original images. The scripts apply cropping and denoising.

This step is not required as the pre-processed images are already available in the `data/datasets` folders. However, if you want to run the scripts, and generate the pre-processed images from the original ones you will have to download the latter from their respective source (see below) and modify the paths in the scripts accordingly.

### Feature extraction
- Generate the Python-based features by executing the `src/scripts/generate_features.py` script.

- Retrieve the LIFEx-generated features by running the `src/scripts/retrieve_lifex_features.py` script. This script retrieves the features from the LIFEx-generated output files contained in the `data\datasets\{dataset}\lifex_raw_features` folder.

In both cases the features will be saved as .csv files in the `output/features/{dataset}` folder, where `{dataset}` indicates the dataset's name. Note that the folders already contain the pre-computed features. Only run the above scripts if you want to compute/retrieve them again.

### Performance estimation of individual feature sets
Execute the `src/scripts/performance_of_single_descriptors.py` to estimate the performance of individual feature sets and generate the ranking. The results will be saved as .csv files in the `output` folder (pre-computed results are already there).

### Performance estimation of ensemble models
Execute the `src/scripts/performance_of_combined_descriptors.py` to estimate the performance of ensemble models. The results will be saved as .csv files in the `output` folder (pre-computed results are already there).

### Utilities
- Use the `src/utilities/generate_charts.py` to generate the charts from the results files.
- Use the `src/utilities/generate_LaTeX_tables.py` to generate LaTeX tables from the results files.

## Dependencies
- [ml_routines](https://github.com/bianconif/ml_routines)
- mpl-ornaments 0.0.4
- numpy 1.24.4
- opencv-python 4.6.0.66
- Pandas 1.4.2
- pyfeats 1.0.1
- pillow 9.1.0
- scikit-image 0.19.3
- scikit-learn 1.4.0
- scipy 1.9.1
- seaborn 0.13.0
- statsmodels 0.14.0
- tabulate 0.8.9
- torch 2.2.0+cu118
- Torchvision 0.17.0+cu118

## Credits {#credits}
- The images contained in the `data/datasets/BrEaST/images` folder are derived from the [Curated benchmark dataset for ultrasound based breast lesion analysis](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/) [1,2] (available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
- The images contained in the `data/datasets/BUID/images` folder are derived from the [Breast Ultrasound Image Database](https://qamebi.com/breast-ultrasound-images-database/) [3-5] (license unspecified).

## License
The code in this repository is available under [GNU General Public License 3.0](https://www.gnu.org/licenses/gpl-3.0.txt). Apart from what specified in [Credits](#credits) all the other material is provided under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## References
1. Pawłowska, A., Ćwierz-Pieńkowska, A., Domalik, A. et al. Curated benchmark dataset for ultrasound based breast lesion analysis. Sci Data 11, 148 (2024). https://doi.org/10.1038/s41597-024-02984-z
2. Pawłowska, A., Ćwierz-Pieńkowska, A., Domalik, A., Jaguś, D., Kasprzak, P., Matkowski, R., Fura, Ł., Nowicki, A., & Zolek, N. (2024). A Curated Benchmark Dataset for Ultrasound Based Breast Lesion Analysis (Breast-Lesions-USG) (Version 1) [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/9WKK-Q141
3. Abbasian Ardakani A., Mohammadi A., Mirza-Aghazadeh-Attari M., Acharya U.R. An open-access breast lesion ultrasound image database‏: Applicable in artificial intelligence studies (2023) Computers in Biology and Medicine, 152, art. no. 106438
4. Homayoun H., Chan W.Y., Kuzan T.Y., Leong W.L., Altintoprak K.M., Mohammadi A., Vijayananthan A., Rahmat K., Leong S.S., Mirza-Aghazadeh-Attari M., Ejtehadifar S., Faeghi F., Acharya U.R., Ardakani A.A. Applications of machine-learning algorithms for prediction of benign and malignant breast lesions using ultrasound radiomics signatures: A multi-center study (2022) Biocybernetics and Biomedical Engineering, 42 (3), pp. 921-933
5. Hamyoon H., Yee Chan W., Mohammadi A., Yusuf Kuzan T., Mirza-Aghazadeh-Attari M., Leong W.L., Murzoglu Altintoprak K., Vijayananthan A., Rahmat K., Ab Mumin N., Sam Leong S., Ejtehadifar S., Faeghi F., Abolghasemi J., Ciaccio E.J., Rajendra Acharya U., Abbasian Ardakani A. Artificial intelligence, BI-RADS evaluation and morphometry: A novel combination to diagnose breast cancer using ultrasonography, results from multi-center cohorts (2022) European Journal of Radiology, 157, art. no. 110591
