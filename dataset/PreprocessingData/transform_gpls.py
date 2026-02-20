import pandas as pd
import os
import traceback

df_total = pd.read_csv('gene_info.csv', sep='\t')
df_total = df_total[['symbols', 'uniprotIds', 'ensemblIds']]
df_total = df_total.drop_duplicates(subset=['symbols', 'uniprotIds', 'ensemblIds'])
print(len(df_total))
standard_gens_symbols = [word.upper() for word in list(df_total['symbols'])]
standard_geneNames = list(df_total['symbols'])
standard_uniprotIds = list(df_total['uniprotIds'])
standard_ensemblIds = list(df_total['ensemblIds'])
case_insensitive_map = {s.casefold(): s for s in standard_gens_symbols}


def is_gene_symbol_column(column, standard_gene_symbols, min_match_count):
    valid_values = column.dropna().drop_duplicates()
    valid_values = valid_values.astype(str).str.upper()
    matching_count = valid_values.isin(standard_gene_symbols).sum()
    return matching_count >= min_match_count


def is_uniprotIds_column(column, standard_uniprotIds, min_match_count):
    valid_values = column.dropna().drop_duplicates()
    matching_count = valid_values.isin(standard_uniprotIds).sum()
    return matching_count >= min_match_count


def is_ensemblIds_column(column, standard_ensemblIds, min_match_count):
    valid_values = column.dropna().drop_duplicates()
    matching_count = valid_values.isin(standard_ensemblIds).sum()
    return matching_count >= min_match_count


# def fuzzy_similarity(word_A, word_B):
#     similarity = fuzz.ratio(word_A, word_B)
#     return similarity


# def is_geneNames_column(column, standard_geneNames, min_match_count):
#     valid_values = column.dropna().drop_duplicates()
#     matching_count = 0 
#     for valid_value in valid_values:
#         for word in standard_geneNames:
#             if fuzzy_similarity(valid_value, word) > 0.9:
#                 matching_count += 1
#     return matching_count >= min_match_count


def filter_column(df, min_match_count=20):
    first_column_name = df.columns[0]
    df.columns.values[0] = 'ID'
    for column in df.columns:
        if is_gene_symbol_column(df[column], standard_gens_symbols, min_match_count):
            if 'symbols' in df.columns:
                df = df.drop(columns=['symbols'])
            if column == 'ID':
                df['symbols'] = df['ID']
            else:
                df[column] = df[column].astype(str).str.upper()
                df = df.rename(columns={column: 'symbols'})
            continue
        if is_uniprotIds_column(df[column], standard_uniprotIds, min_match_count):
            if 'uniprotIds' in df.columns:
                df = df.drop(columns=['uniprotIds'])
            if column == 'ID':
                df['uniprotIds'] = df['ID']
            else:
                df = df.rename(columns={column: 'uniprotIds'})
            continue

        if is_ensemblIds_column(df[column], standard_ensemblIds, min_match_count):
            if 'ensemblIds' in df.columns:
                df = df.drop(columns=['ensemblIds'])
            if column == 'ID':
                df['ensemblIds'] = df['ID']
            else:
                df = df.rename(columns={column: 'ensemblIds'})
            continue

    if 'symbols' in df.columns or 'uniprotIds' in df.columns  or 'ensemblIds' in df.columns:
        non_none_columns = [column for column in ['symbols', 'uniprotIds', 'ensemblIds'] if column in df.columns]
        df = df[['ID']+non_none_columns]
        print(df.columns)
        df = df.drop_duplicates(subset=non_none_columns)
        df = pd.merge(df, df_total, on=non_none_columns, how='left')
        print(df.columns)
        df = df[['ID', 'symbols', 'uniprotIds', 'ensemblIds']]
        # df.columns = ['ID', 'symbols', 'uniprotIds', 'ensemblIds']
        return df


for index, file in enumerate(os.listdir('./gpls')):

    gpl_input_filepath = 'gpls/' + file
    gpl_processed_filepath = 'gpls_trans/' + file
    if os.path.exists(gpl_processed_filepath):
        continue
    try:
        df_gpl = pd.read_csv(gpl_input_filepath)
        rst_df = filter_column(df_gpl)
        if rst_df is not None:
            rst_df.to_csv(gpl_processed_filepath, sep='\t')
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        traceback.print_exc()