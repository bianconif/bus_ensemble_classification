import matplotlib.pyplot as plt
from mpl_ornaments.titles import set_title_and_subtitle
import os
import pandas as pd
import seaborn as sns

from common import charts_folder, complete_res_comb_file

if not os.path.isdir(charts_folder):
    os.makedirs(charts_folder)
    
df_combined = pd.read_csv(complete_res_comb_file)

#===================================================================
#=========== Strip-plots/box-plots by combined feature set =========
#===================================================================
fig, ax = plt.subplots(nrows=1, ncols=1)

sns.boxplot(data=df_combined, y='Acc.', x='Descriptor', 
            hue='Descriptor', whis=1.5, showfliers=False, 
            boxprops={'alpha': 0.4}, ax=ax)

sns.stripplot(data=df_combined, y='Acc.', x='Descriptor',
              hue='Fusion method', jitter=0.2, edgecolor='auto', 
              linewidth=1, ax=ax)

#sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

title = 'Performance of ensemble models'
subtitle = 'Grouped by aggregation of feature sets' 
set_title_and_subtitle(fig=fig, title=title, subtitle=subtitle,
                       alignment='left', h_offset=25)

ax.set(**{'xlabel': None, 'ylabel': 'Accuracy (%)'})
ax.tick_params(axis='x', labelrotation=90)
ax.grid(visible=True, axis='y')
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.set_ylim(bottom=60, top=90)

fig.savefig(f'{charts_folder}/by-aggregation-of-feature-sets.png', 
            dpi=450, bbox_inches='tight')
#===================================================================
#===================================================================
#===================================================================

#===================================================================
#============== Strip-plots/box-plots by fusion method =============
#===================================================================
fig, ax = plt.subplots(nrows=1, ncols=1)

sns.boxplot(data=df_combined, y='Acc.', x='Fusion method', 
            hue='Fusion method', whis=1.5, showfliers=False, 
            boxprops={'alpha': 0.4}, ax=ax)

sns.stripplot(data=df_combined, y='Acc.', x='Fusion method',
              hue='Descriptor', jitter=0.2, edgecolor='auto', 
              linewidth=1, ax=ax)

#sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

title = 'Performance of ensemble models'
subtitle = 'Grouped by fusion method' 
set_title_and_subtitle(fig=fig, title=title, subtitle=subtitle,
                       alignment='left', h_offset=25)

ax.set(**{'xlabel': None, 'ylabel': 'Accuracy (%)'})
ax.tick_params(axis='x', labelrotation=90)
ax.set_ylim(bottom=60, top=90)
ax.grid(visible=True, axis='y')
ax.spines[['top', 'right', 'left']].set_visible(False)

fig.savefig(f'{charts_folder}/by-fusion-method.png', 
            dpi=450, bbox_inches='tight')
#===================================================================
#===================================================================
#===================================================================