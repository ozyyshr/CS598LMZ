import pandas as pd


datasets = ['testcasegen']
splits = ['train', 'test']


# for dataset in datasets:
#     for split in splits:
#         path = f'data/sql/{dataset}/{split}.parquet'
#         df = pd.read_parquet(path)
#         print(f"dataset: {dataset}, split: {split}, Loaded {len(df)} rows")
#         if split == 'test' and dataset == 'bird':
#             print(df.head(1)['prompt'][0][0]['content'])
#             break


path = f'data/testcasegen/train.parquet'
df = pd.read_parquet(path)
print(df.head(1)['prompt'][0][0]['content'])
# print(df.head(5))

# for i in range(len(df)):
#     # print(df.iloc[i]['prompt'])
#     print(df.iloc[i]['extra_info']['db_path'])
#     print('--------------------------------')
#     break
