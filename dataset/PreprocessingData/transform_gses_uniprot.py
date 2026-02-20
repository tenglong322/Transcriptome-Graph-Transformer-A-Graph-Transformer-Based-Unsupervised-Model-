import pandas as pd
import os


df = pd.read_csv('gene_info.csv', sep='\t')
df_cleaned = df.dropna(subset=['symbols', 'uniprotIds'])
standard_gens_symbols = [word.upper() for word in list(df_cleaned['symbols'])]
standard_uniprotIds = list(df_cleaned['uniprotIds'])
gene_uniprot_dict = dict(zip(standard_gens_symbols, standard_uniprotIds))


df_ppi = pd.read_csv('ppi_processed.txt', sep='\t') 
ppi_ls = list(set(list(df_ppi['Interactor1'])+list(df_ppi['Interactor2'])))


for index,file in enumerate(os.listdir('./gses_trans')):
    file_path = 'gses_trans/'+file
    df_gse=pd.read_csv(file_path, sep='\t')
    if len(df_gse) > 0:
        gse_name = file.split('.')[0]
        gse_trans_path = 'gses_uniprot/'+gse_name+'.csv'
        df_gse['Number'] = df_gse['Number'].map(gene_uniprot_dict)
        df_gse = df_gse.dropna(subset=['Number'])
        df_gse = df_gse.drop_duplicates(subset=['Number'])
        df_gse = df_gse[df_gse['Number'].isin(ppi_ls)]
        if len(df_gse) > 3:
            df_gse.to_csv(gse_trans_path, index=False, sep='\t')
            










