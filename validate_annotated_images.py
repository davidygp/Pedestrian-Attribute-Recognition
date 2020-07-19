import pandas as pd

df = pd.read_csv("./program_annotated_zero_one_pa100k.csv")

df = df.drop(df.columns[0], axis=1)

df = df.reindex(sorted(df.columns), axis=1)

df = df.loc[df['image_id'] > '075000.jpg']

working_df = df.iloc[0: 1000, :]

headers = df.columns.to_list()

working_df.to_csv('pa100k_75001_76000.csv')

pass
