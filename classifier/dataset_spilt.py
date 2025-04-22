# multiclass_onehot_split.py

import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def main():
    # —— 1. 读原始数据 ——  
    df = pd.read_excel('spacerdl_fun_sim.xlsx')  # 按需改成 pd.read_csv(...)
    
    # —— 2. 构造联合标签 path_label ——  
    # LEVEL_COLS = ['Level1','Level2','Level3','Level4']
    LEVEL_COLS = ['Label']
    df['path_label'] = (
        df[LEVEL_COLS]
        .fillna('')
        .agg('/'.join, axis=1)
        .str.strip('/')
    )
    
    # —— 3. one‑hot 编码 ——  
    onehot = pd.get_dummies(df['path_label'])
    df_onehot = pd.concat([
        df[['Requirements Description']], 
        onehot
    ], axis=1)
    
    # —— 4. 供分层的 y ——  
    y = df['path_label']
    
    # —— 5. 外层：10 折 85%/15% 分层划分 ——  
    outer = StratifiedShuffleSplit(
        n_splits=10,
        train_size=0.85,
        test_size=0.15,
        random_state=42
    )
    
    base_dir = 'splits_sim'
    for fold, (idx_trval, idx_test) in enumerate(outer.split(df_onehot, y), start=1):
        # 每折创建自己的文件夹
        folder = os.path.join(base_dir, f'fold_{fold}')
        os.makedirs(folder, exist_ok=True)
        
        # 划分 train+val 和 test
        df_trval = df_onehot.iloc[idx_trval].reset_index(drop=True)
        y_trval  = y.iloc[idx_trval].reset_index(drop=True)
        df_test  = df_onehot.iloc[idx_test].reset_index(drop=True)
        
        # —— 6. 内层：在 trainval 上做 70/85 vs 15/85 分层 ——  
        train_ratio = 0.70 / 0.85
        val_ratio   = 0.15 / 0.85
        inner = StratifiedShuffleSplit(
            n_splits=1,
            train_size=train_ratio,
            test_size=val_ratio,
            random_state=100 + fold
        )
        idx_train, idx_val = next(inner.split(df_trval, y_trval))
        df_train = df_trval.iloc[idx_train].reset_index(drop=True)
        df_val   = df_trval.iloc[idx_val].reset_index(drop=True)
        
        # —— 7. 保存到各自文件夹 ——  
        df_train.to_csv(os.path.join(folder, 'train.csv'), index=False)
        df_val.to_csv(  os.path.join(folder, 'val.csv'),   index=False)
        df_test.to_csv( os.path.join(folder, 'test.csv'),  index=False)
        
        print(f'Fold {fold}: '
              f'train={len(df_train):3d}, '
              f'val={len(df_val):3d}, '
              f'test={len(df_test):3d}')

if __name__ == '__main__':
    main()
