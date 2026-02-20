import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

input_folder = "gses_uniprot"
output_folder = "gses_uniprot_scaled"
os.makedirs(output_folder, exist_ok=True)
low_percentile = 1
high_percentile = 99
correlation_threshold = 0.95  

for file in os.listdir(input_folder):
    try:
        if not file.endswith(".csv") and not file.endswith(".txt"):
            continue
        file_path = os.path.join(input_folder, file)
        sep = '\t'
        df = pd.read_csv(file_path, sep=sep)
        if "Number" not in df.columns:
            continue
        df.set_index("Number", inplace=True)
        df = df.fillna(0)

        corr_matrix = df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
        if to_drop:
            df = df.drop(columns=to_drop)
            print(f"Dropped {len(to_drop)} ")

        min_val = df.min().min()
        df_shifted = df - min_val if min_val < 0 else df.copy()
        df_log = np.log1p(df_shifted)

        low_val = np.percentile(df_log.values, low_percentile)
        high_val = np.percentile(df_log.values, high_percentile)
        df_clipped = df_log.clip(lower=low_val, upper=high_val)

        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled = pd.DataFrame(scaler.fit_transform(df_clipped), index=df_clipped.index, columns=df_clipped.columns)

        out_path = os.path.join(output_folder, file)
        df_scaled.to_csv(out_path, sep='\t')

    except Exception as e:
        print(f"Error processing {file}: {e}")
        continue
