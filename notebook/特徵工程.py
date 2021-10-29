import pandas as pd
import time
from tqdm import tqdm
import gc
import warnings 
warnings.filterwarnings('ignore')

train_data = pd.read_feather('../data/train_handle_nan.feather')
test_data = pd.read_feather('../data/test_handle_nan.feather')

feature_columns = list(set(train_data.columns.tolist()) - set(['chid','shop_tag']))
categorical_features = ['masts','educd','trdtp','naty','poscd','cuorg','gender_code','age','primary_card']
numerical_features = list(set(feature_columns)-set(categorical_features))
train_df = train_data
sample_submit = test_data

# 從source_df抓對應的chid算一些統計資訊當作特徵併入target_df裡面
def feature_engineer(target_df,source_df):
    for idx in tqdm(target_df.index):
        # 取出這一條row的chid
        chid = target_df.loc[idx,'chid']
        # 從source_df取出所有屬於這個chid的row
        group_by_chid = source_df.loc[source_df.chid==chid]
        # 數值特徵(numerical_features)統計資訊
        for col in numerical_features: 
            target_df.loc[idx,f"mean_of_{col}"] = group_by_chid.loc[:,col].mean()
            target_df.loc[idx,f"max_of_{col}"] = group_by_chid.loc[:,col].max()
            target_df.loc[idx,f"min_of_{col}"] = group_by_chid.loc[:,col].min()
        # 類別特徵(categorical_features)統計資訊
        for col in categorical_features:
            # 1.nunique (這個欄位有多少種不一樣的數值)
            target_df.loc[idx,f"nunique_of_{col}"] = group_by_chid.loc[:,col].nunique()
            # 2.norm_count (這個欄位在每一種數值上出現的次數(count)除以總數(normalize)等於取平均(mean))
            cat_feat = pd.get_dummies(group_by_chid.loc[:,col],columns=col) # one_hot 的 np.array
            cat_feat = pd.DataFrame(cat_feat.mean(axis=0)).T # 轉成DataFrame取平均
            cat_feat.index = [idx] # 處理DataFrame格式
            cat_feat.columns = [ "norm_count_" + col + "_" + str(i) for i in cat_feat.columns] # 處理DataFrame格式
            try:
                target_df = target_df.join(cat_feat) # 合併DataFrame
            except:
                target_df.loc[idx,cat_feat.columns] = cat_feat.values.reshape(-1)
    return target_df

sample_submit_feature = feature_engineer(sample_submit,train_data)
train_data_feature = feature_engineer(train_data,train_data)

sample_submit_new_features = list(set(sample_submit_feature.columns)-set(['chid','top1','top2','top3']))
common_features = list(set(sample_submit_new_features)&set(train_data_feature.columns))

# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

train_miss = missing_values_table(train_data_feature)
test_miss = missing_values_table(sample_submit_feature)

norm_count_col = [i if 'norm' in i else 'None' for i in train_data_feature.columns.tolist()+sample_submit_feature.columns.tolist()]
norm_count_col = set(norm_count_col)-set(['None'])

import numpy as np
# 類別型特徵norm_count代表 groupby chid 對應某一個類別欄位 在某一種類別上出現的概率
# 如果為1.0 代表這個 chid 在 某一個類別欄位 其中一種類別上出現的概率為100% 反之為0%
# 如果是缺失值代表這個chid的某一項類別欄位沒有出現過該種類 既然沒有出現過 所以概率= 0%
set_value = 0.0
for col in set(norm_count_col)&set(train_data_feature.columns):
    train_data_feature[col].fillna(set_value,inplace=True)
for col in set(norm_count_col)&set(sample_submit_feature.columns): 
    sample_submit_feature[col].fillna(set_value,inplace=True)

train_miss = missing_values_table(train_data_feature)
test_miss = missing_values_table(sample_submit_feature)

print(set([ i.split('_')[0] for i in train_miss.index]))
print(set([ i.split('_')[0] for i in test_miss.index]))

# 數值型特徵欄位補植
# 數值型特徵代表 groupby chid 在某一個特徵上的,最大,最小,平均值 等等統計數值
# 數值型特徵的"大小"是有意義的會影響激活函數,因此如果缺值補最小值或最大值都太過於極端,因此補"中位數" (平均值會受到極端值影響)
for col in train_miss.index:
    train_data_feature[col] = train_data_feature[col].fillna(train_data_feature[col].median())
for col in test_miss.index:
    sample_submit_feature[col] = sample_submit_feature[col].fillna(sample_submit_feature[col].median())

print(train_data_feature.shape)
print(sample_submit_feature.shape)

train_data_feature = train_data_feature[common_features+['shop_tag']]
sample_submit_feature = sample_submit_feature[common_features]

print(train_data_feature.shape)
print(sample_submit_feature.shape)
print(train_data_feature.isnull().sum().sum())
print(sample_submit_feature.isnull().sum().sum())

train_data_feature.reset_index().to_feather('../data/train_data_feature.feather')
sample_submit_feature.reset_index().to_feather('../data/sample_submit_feature.feather')
print('DONE!')