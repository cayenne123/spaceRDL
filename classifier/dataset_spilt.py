# multiclass_onehot_split.py

import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# def main():
#     # —— 1. 读原始数据 ——  
#     df = pd.read_excel('D:/LLM-code/spaceRDL/classifier/spacerdl_fun.xlsx')  # 按需改成 pd.read_csv(...)
    
#     # —— 2. 构造联合标签 path_label ——  
#     LEVEL_COLS = ['Level1','Level2','Level3','Level4']
#     # LEVEL_COLS = ['Label']
#     df['path_label'] = (
#         df[LEVEL_COLS]
#         .fillna('')
#         .agg('/'.join, axis=1)
#         .str.strip('/')
#     )
    
#     # —— 3. one‑hot 编码 ——  
#     onehot = pd.get_dummies(df['path_label'])
#     df_onehot = pd.concat([
#         df[['Requirements Description']], 
#         onehot
#     ], axis=1)
    
#     # —— 4. 供分层的 y ——  
#     y = df['path_label']
    
#     # —— 5. 外层：10 折 85%/15% 分层划分 ——  
#     outer = StratifiedShuffleSplit(
#         n_splits=10,
#         train_size=0.85,
#         test_size=0.15,
#         random_state=42
#     )
    
#     base_dir = 'classifier/splits'
#     for fold, (idx_trval, idx_test) in enumerate(outer.split(df_onehot, y), start=1):
#         # 每折创建自己的文件夹
#         folder = os.path.join(base_dir, f'fold_{fold}')
#         os.makedirs(folder, exist_ok=True)
        
#         # 划分 train+val 和 test
#         df_trval = df_onehot.iloc[idx_trval].reset_index(drop=True)
#         y_trval  = y.iloc[idx_trval].reset_index(drop=True)
#         df_test  = df_onehot.iloc[idx_test].reset_index(drop=True)
        
#         # —— 6. 内层：在 trainval 上做 70/85 vs 15/85 分层 ——  
#         train_ratio = 0.84 / 0.85
#         val_ratio   = 0.01 / 0.85
#         inner = StratifiedShuffleSplit(
#             n_splits=1,
#             train_size=train_ratio,
#             test_size=val_ratio,
#             random_state=100 + fold
#         )
#         idx_train, idx_val = next(inner.split(df_trval, y_trval))
#         df_train = df_trval.iloc[idx_train].reset_index(drop=True)
#         df_val   = df_trval.iloc[idx_val].reset_index(drop=True)
        
#         # —— 7. 保存到各自文件夹 ——  
#         df_train.to_excel(os.path.join(folder, 'train.xlsx'), index=False)
#         df_val.to_excel(  os.path.join(folder, 'val.xlsx'),   index=False)
#         df_test.to_excel( os.path.join(folder, 'test.xlsx'),  index=False)
        
#         print(f'Fold {fold}: '
#               f'train={len(df_train):3d}, '
#               f'val={len(df_val):3d}, '
#               f'test={len(df_test):3d}')

# multiclass_onehot_split_802.py
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def main():
    # —— 1. 读原始数据 ——  
    df = pd.read_excel(
        'D:/LLM-code/spaceRDL/dataset/dataset1338-已标注.xlsx',
        engine='openpyxl'          # 避免因本地默认解析器不同而报错
    )

    # —— 2. 构造联合标签 path_label ——  
    LEVEL_COLS = ['pattern_1', 'pattern_2']
    df['path_label'] = (
        df[LEVEL_COLS]
        .fillna('')
        .agg('/'.join, axis=1)
        .str.strip('/')            # 去掉可能的首尾 /
    )

    # —— 3. One-Hot 编码 ——  
    onehot = pd.get_dummies(df['path_label']).astype(int)     # ← 直接转成 0/1
    df_onehot = pd.concat([df[['description']], onehot], axis=1)

    # —— 4. 单次 8/2 分层拆分 ——  
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=0.8,
        test_size=0.2,
        random_state=42
    )
    
    # 1. 统计每个 label 出现的次数
    label_counts = df['path_label'].value_counts()

    # 2. 筛选出只出现一次的 labels
    rare_labels = label_counts[label_counts == 1].index.tolist()

    # 3. 输出它们
    print(f"Rare labels (出现次数=1):\n{rare_labels}")
    df_rare = df[df['path_label'].isin(rare_labels)]
    print(df_rare[['description', 'path_label']])

    idx_train, idx_test = next(splitter.split(df_onehot, df['path_label']))
    df_train = df_onehot.iloc[idx_train].reset_index(drop=True)
    df_test  = df_onehot.iloc[idx_test].reset_index(drop=True)

    # —— 5. 保存 ——  
    out_dir = 'classifier/split82'
    os.makedirs(out_dir, exist_ok=True)
    df_train.to_excel(os.path.join(out_dir, 'train.xlsx'), index=False)
    df_test.to_excel(os.path.join(out_dir,  'test.xlsx'),  index=False)

    print(f'Done! train={len(df_train)}, test={len(df_test)}')


if __name__ == '__main__':
    main()
