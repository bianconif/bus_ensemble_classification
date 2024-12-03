import os
import pandas as pd

from common import best_res_single, latex_folder, texture_descriptors,\
     cnn_descriptors

def latex_results_table_by_descriptor(df_results, 
                                      out_file,
                                      descriptor_col_in='Descriptor',
                                      acc_col_in='Acc.',
                                      acc_lci_col_in='Acc. CI_l',
                                      acc_uci_col_in='Acc. CI_u',
                                      sens_col_in='Sens.',
                                      spec_col_in='Spec.',
                                      train_col_in='Train',
                                      test_col_in='Test'):
    """LaTeX table showing the results by descriptor
    
    Parameters
    ----------
    out_file: str
        Relative or absolute path the LaTeX output file.
    df_results: pd.DataFrame
        The dataframe containing the results.
    descriptor_col: str
        Name of the column storing the name of the descriptor.
    acc_col_in: str
        Name of the column storing the accuracy value.
    acc_lci_col_in: str
        Name of the column storing the lower bound of the CI for accuracy.
    acc_uci_col_in: str
        Name of the column storing the upper bound of the CI for accuracy.
    sens_col_in: str
        Name of the column storing the sensitivity value.
    spec_col_in: str
        Name of the column storing the specificity value.
    train_col_in: str
        Name of the column storing the name of the train dataset.
    test_col_in: str
        Name of the column storing the name of the test dataset.
        
    Returns
    -------
    None
    """
    
    fp_out_file = open(out_file, 'w')
    
    #Header
    fp_out_file.write('\\begin{tabular}{lllccc}\\\\ \n')
    fp_out_file.write('\\toprule\n')
    fp_out_file.write(f'{descriptor_col_in} & {train_col_in} &')
    fp_out_file.write(f'{test_col_in} & {acc_col_in} & {sens_col_in} &')
    fp_out_file.write(f'{spec_col_in}\\\\ \n')
    fp_out_file.write('\\midrule\n')
    
    #Body
    descriptors_grouped = df_results.groupby(by=descriptor_col_in)
    for idx_descriptor_grp, (descriptor_name, descriptor_grp) in\
        enumerate(descriptors_grouped):
        
        for idx_train_test_grp, (_, train_test_grp) in\
            enumerate(descriptor_grp.groupby(by=[train_col_in, test_col_in])):
            
            if idx_train_test_grp == 0:
                fp_out_file.write(f'\\multirow{{{descriptor_grp.shape[0]}}}{{*}}{{{descriptor_name}}} & ')
            else:
                fp_out_file.write(f' & ')
            
            train = train_test_grp.iloc[0][train_col_in]
            test = train_test_grp.iloc[0][test_col_in]
            acc = train_test_grp.iloc[0][acc_col_in]
            acc_ci_l = train_test_grp.iloc[0][acc_lci_col_in]
            acc_ci_u = train_test_grp.iloc[0][acc_uci_col_in]
            sens = train_test_grp.iloc[0][sens_col_in]
            spec = train_test_grp.iloc[0][spec_col_in]
            acc_ci_str = f'[{acc_ci_l:3.2f}--{acc_ci_u:3.2f}]'
                
            fp_out_file.write(f'{train} & ')
            fp_out_file.write(f'{test} & ')
            fp_out_file.write(f'{acc:3.1f} {acc_ci_str} & ')
            fp_out_file.write(f'{sens:3.1f} & ')
            fp_out_file.write(f'{spec:3.1f}\\\\ \n')
        
        if idx_descriptor_grp != (descriptors_grouped.ngroups - 1):   
            fp_out_file.write('\\midrule\n')
        else:
            fp_out_file.write('\\bottomrule\n')
    
    #Footer
    fp_out_file.write('\\end{tabular}\n')
    
    fp_out_file.close()
    
descriptor_col_in = 'Descriptor'

if not os.path.isdir(latex_folder):
    os.makedirs(latex_folder)

    
#Read the results
df_best_res_single = pd.read_csv(best_res_single)
    
#Results table with morphological and intensity-based features
df_slice = df_best_res_single[df_best_res_single[descriptor_col_in].\
                              isin(['Morphological', 'intensity-based',
                                    'intensity-histogram'])]
latex_results_table_by_descriptor(
    df_results=df_slice, 
    out_file=f'{latex_folder}/morphological+intensity-based.tex',
    descriptor_col_in=descriptor_col_in
)

#Results table with texture features
df_slice = df_best_res_single[df_best_res_single[descriptor_col_in].\
                              isin(texture_descriptors.keys())]
latex_results_table_by_descriptor(
    df_results=df_slice, 
    out_file=f'{latex_folder}/texture.tex',
    descriptor_col_in=descriptor_col_in
)

#Results table with CNN-based features
df_slice = df_best_res_single[df_best_res_single[descriptor_col_in].\
                              isin(cnn_descriptors.keys())]
latex_results_table_by_descriptor(
    df_results=df_slice, 
    out_file=f'{latex_folder}/cnn.tex',
    descriptor_col_in=descriptor_col_in
)