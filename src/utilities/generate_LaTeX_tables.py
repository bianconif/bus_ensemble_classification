from itertools import product
import os
import pandas as pd

from common import best_res_single_file, best_res_combined_file,\
     latex_folder, texture_descriptors,\
     cnn_descriptors, ranking_single_file

def latex_ranking_table(df_ranking, out_file,
                        descriptor_col_in='Descriptor',
                        wins_col_in='Wins',
                        losses_col_in='Losses',
                        ties_col_in='Ties',
                        points_col_in='Points',
                        rank_col_in='Rank'
                        ):
    """LaTeX table showing the ranking of individual descriptors
    
    Parameters
    ----------
    df_results: pd.DataFrame
        The dataframe containing the results.
    out_file: str
        Relative or absolute path the LaTeX output file.
    descriptor_col_in: str
        Name of the column storing the name of the descriptor.
    wins_col_in: str
        Name of the column storing the number of wins.
    losses_col_in: str
        Name of the column storing the number of losses.
    ties_col_in: str
        Name of the column storing the number of ties.
    points_col_in: str
        Name of the column storing the number of points.
    rank_col_in: str
        Name of the column storing the rank.
        
    Returns
    -------
    None
    """ 
    
    fp_out_file = open(out_file, 'w')
    
    #Header
    fp_out_file.write('\\begin{tabular}{llllll}\\\\ \n')
    fp_out_file.write('\\toprule\n')
    fp_out_file.write(f'{descriptor_col_in} & {wins_col_in} & ')
    fp_out_file.write(f'{losses_col_in} & {ties_col_in} & {points_col_in} & ')
    fp_out_file.write(f'{rank_col_in}\\\\ \n')
    fp_out_file.write('\\midrule\n')    
    
    #Body
    for _, row in df_ranking.iterrows():
        
        descriptor, wins, losses, ties, points, rank =\
            row[[
                descriptor_col_in, wins_col_in, losses_col_in, 
                ties_col_in, points_col_in, rank_col_in
            ]]
        
        fp_out_file.write(f'{descriptor} & {wins} & {losses} & {ties} & '
                          f'{points:d} & {rank}\\\\ \n')
        
    #Footer
    fp_out_file.write('\\bottomrule\n')
    fp_out_file.write('\\end{tabular}\n')
    
    
    fp_out_file.close()

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
    df_results: pd.DataFrame
        The dataframe containing the results.
    out_file: str
        Relative or absolute path the LaTeX output file.
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
    
def latex_results_table_single_vs_combined(
    df_best_single,
    df_best_combined,
    out_file,
    fusion_method_col_in = 'Fusion method',
    descriptor_col_in='Descriptor',
    acc_col_in='Acc.',
    acc_lci_col_in='Acc. CI_l',
    acc_uci_col_in='Acc. CI_u',
    sens_col_in='Sens.',
    spec_col_in='Spec.',
    train_col_in='Train',
    test_col_in='Test'):
    """
    Parameters
    ----------
    df_best_single: pd.DataFrame
        The dataframe containing the best results for each single
        descriptor.
    df_best_combined: pd.DataFrame
        The dataframe containing the best results for each combined
        descriptor.
    out_file: str
        Relative or absolute path the LaTeX output file.
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
    experimental_conditions = product(
        df_best_single[train_col_in].unique(),
        df_best_single[test_col_in].unique()
    )
    
    fp_out_file = open(out_file, 'w')
    
    #Header
    fp_out_file.write('\\begin{tabular}{llllccc}\\\\ \n')
    fp_out_file.write('\\toprule\n')
    fp_out_file.write(f'{descriptor_col_in} & Fusion method & {train_col_in} &')
    fp_out_file.write(f'{test_col_in} & {acc_col_in} & {sens_col_in} &')
    fp_out_file.write(f'{spec_col_in}\\\\ \n')
    fp_out_file.write('\\midrule\n')
    
    for experimental_condition in experimental_conditions:
        
        df_best_single_slice = df_best_single.loc[
            (df_best_single[train_col_in] == experimental_condition[0]) &
            (df_best_single[test_col_in] == experimental_condition[1])
        ]
        df_best_single_slice.sort_values(by=acc_col_in, ascending=False,
                                         inplace=True)
        df_best_single_slice = df_best_single_slice.iloc[0]
        
        descriptor = df_best_single_slice[descriptor_col_in]
        acc = df_best_single_slice[acc_col_in]
        acc_ci_l = df_best_single_slice[acc_lci_col_in]
        acc_ci_u = df_best_single_slice[acc_uci_col_in]
        sens = df_best_single_slice[sens_col_in]
        spec = df_best_single_slice[spec_col_in]
        acc_ci_str = f'[{acc_ci_l:3.2f}--{acc_ci_u:3.2f}]'
        
        fp_out_file.write(f'{descriptor} & & ')
        fp_out_file.write(f'{experimental_condition[0]} & ')
        fp_out_file.write(f'{experimental_condition[1]} & ')
        fp_out_file.write(f'{acc:3.1f} {acc_ci_str} & ')
        fp_out_file.write(f'{sens:3.1f} & ')
        fp_out_file.write(f'{spec:3.1f}\\\\ \n')
        fp_out_file.write('\\midrule\n')
        
        df_best_combined_slice = df_best_combined.loc[
            (df_best_combined[train_col_in] == experimental_condition[0]) &
            (df_best_combined[test_col_in] == experimental_condition[1])
        ]
        df_best_combined_slice.sort_values(by=acc_col_in, ascending=False,
                                         inplace=True)
        df_best_combined_slice = df_best_combined_slice.iloc[0:4]
        
        for _, row in df_best_combined_slice.iterrows():
            
            descriptor = row[descriptor_col_in]
            fusion_method = row[fusion_method_col_in]
            acc = row[acc_col_in]
            acc_ci_l = row[acc_lci_col_in]
            acc_ci_u = row[acc_uci_col_in]
            sens = row[sens_col_in]
            spec = row[spec_col_in]
            acc_ci_str = f'[{acc_ci_l:3.2f}--{acc_ci_u:3.2f}]'
            
            fp_out_file.write(f'{descriptor} & ')
            fp_out_file.write(f'{fusion_method} & ')
            fp_out_file.write(f'{experimental_condition[0]} & ')
            fp_out_file.write(f'{experimental_condition[1]} & ')
            fp_out_file.write(f'{acc:3.1f} {acc_ci_str} & ')
            fp_out_file.write(f'{sens:3.1f} & ')
            fp_out_file.write(f'{spec:3.1f}\\\\ \n') 
        
        fp_out_file.write('\\midrule\n')
        fp_out_file.write('\\midrule\n')
    
    #Footer
    fp_out_file.write('\\bottomrule\n')
    fp_out_file.write('\\end{tabular}\n')
    
    fp_out_file.close()
    

if not os.path.isdir(latex_folder):
    os.makedirs(latex_folder)

descriptor_col_in = 'Descriptor'
    
#Read the results
df_best_res_single = pd.read_csv(best_res_single_file)
df_best_res_combined = pd.read_csv(best_res_combined_file)

latex_results_table_single_vs_combined(
    df_best_single=df_best_res_single,
    df_best_combined=df_best_res_combined,
    out_file=f'{latex_folder}/single-vs-combined.tex')
    
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

#Ranking table
df_ranking = pd.read_csv(ranking_single_file)
latex_ranking_table(
    df_ranking, 
    out_file=f'{latex_folder}/ranking.tex'
)