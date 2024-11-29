#Run round-robin for establishing a ranking
from itertools import combinations, product
import pandas as pd

df_pruned = pd.read_csv('best-performance-single-descriptors.csv')
subset = ['Descriptor', 'Train', 'Test']
df_pruned = df_pruned.drop_duplicates(subset=subset)
df_pruned.set_index(keys=subset, inplace=True)

df_round_robin = pd.DataFrame(
    index=df_pruned.index.get_level_values('Descriptor').unique(),
    columns=['Wins', 'Losses', 'Ties'], 
)
df_round_robin.fillna(0, inplace=True)

pairing_table = combinations(df_round_robin.index, r=2)
for home, visitor in pairing_table:
    for train, test in product(
        df_pruned.index.get_level_values('Train').unique(), 
        df_pruned.index.get_level_values('Test').unique()
    ):
        
        home_record=df_pruned.loc[home, train, test]
        visitor_record=df_pruned.loc[visitor, train, test]
        
        #Assign the points
        if home_record['Acc. CI_l'] > visitor_record['Acc. CI_u']:
            #Home wins
            df_round_robin.loc[home]['Wins'] += 1
            df_round_robin.loc[visitor]['Losses'] += 1
        elif visitor_record['Acc. CI_l'] > home_record['Acc. CI_u']:
            #Visitor wins
            df_round_robin.loc[visitor]['Wins'] += 1
            df_round_robin.loc[home]['Losses'] += 1
        else:
            #Tie
            df_round_robin.loc[visitor]['Ties'] += 1
            df_round_robin.loc[home]['Ties'] += 1  
            
df_round_robin['Points'] = (-1 * df_round_robin['Losses']) +\
                           (0 * df_round_robin['Ties']) +\
                           (1 * df_round_robin['Wins'])
df_round_robin['Rank'] = df_round_robin['Points'].rank()
df_round_robin.sort_values(by='Rank', ascending=False, inplace=True)