import pandas as pd
import os
import shutil

df = pd.read_csv('../../data/replace_all.csv', error_bad_lines=False, index_col=False)
series_df = df['label'].unique()
for lb in series_df:
    file_path = os.path.join('../../data/all/', lb)
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    for row in df.iterrows():
        print(row[1]['path'])
        if lb == row[1]['label']:
            if os.path.exists(row[1]['path']):
                file_name = os.path.split(row[1]['path'])[-1]
                try:
                    shutil.copy(row[1]['path'], os.path.join(file_path,file_name))
                except:
                    pass

# df = pd.read_csv('./train.csv')
# series_df = df['label'].unique()
# for lb in series_df:
#     file_path = './files/' + lb
#     if not os.path.exists(file_path):
#         os.mkdir(file_path)
#     for row in df.iterrows():
#         print(row[1]['path'])
#         if lb == row[1]['label']:
#             if os.path.exists(row[1]['path']):
#                 file_name = row[1]['path'].split('/')[-1]
#                 shutil.copy(row[1]['path'], file_path + "/" + file_name)



# df = pd.read_csv('./validation.csv')
# series_df = df['label'].unique()
# for lb in series_df:
#     file_path = './data/validate/' + lb
#     if not os.path.exists(file_path):
#         os.mkdir(file_path)
#     for row in df.iterrows():
#         print(row[1]['path'])
#         if lb == row[1]['label']:
#             if os.path.exists(row[1]['path']):
#                 file_name = row[1]['path'].split('/')[-1]
#                 shutil.copy(row[1]['path'], file_path + "/" + file_name)
pass
