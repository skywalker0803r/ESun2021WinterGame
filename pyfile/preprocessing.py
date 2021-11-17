import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import gc
import logging
import coloredlogs

# 顯示設定
gc.collect()
tqdm.pandas()
log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

# 超參數設定
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

args = dotdict({
    'pca_n':84,
    'knn_k':42,
    'knn_leaf_size':300,
    'knn_p':2,
    'start_dt':12,
})
log.info(f'setting args is :{args}')

# LOAD DATA
log.info('start load data...')
df = pd.read_feather('../data/2021玉山人工智慧公開挑戰賽冬季賽訓練資料集.feather')
df = df.loc[df.dt>=args.start_dt,:] #取大於start_dt的資料

# FILL MISS VALUE
log.info('start fill miss values')
mean_fill_col = ['slam'] # 數值型用中位數補值(避免均值被離群值拉偏)
cat_fill_col = ['gender_code','age','trdtp','educd','masts','naty','poscd','cuorg'] # 類別型用一個-999補值
for i in tqdm(mean_fill_col):
    df[i] = df[i].fillna(df[i].median())
for i in tqdm(cat_fill_col):
    df[i] = df[i].fillna(-999)
assert df.isnull().sum().sum() == 0

log.info("start create_grouby_chid_df...")
# 以下根據chid做groupby操作,並且因應各種不同類型的欄位作客製化統計運算(沿時間維度取mean之類的)
def create_groupby_chid_df(df):
    
    # 1 groupby('chid')根據時間維度取最後一筆(最近期資料)
    final_col = ['age','primary_card','trdtp','educd','gender_code','masts','poscd','naty','cuorg'] 
    var1 = df.groupby('chid')[final_col].agg(lambda x:x.values[-1]) # 最後一筆 
    
    # 2 groupby('chid')根據時間維度算平均百分比
    mean_col = df.columns[df.columns.str.contains('pct')]
    var2 = df.groupby('chid')[mean_col].agg('mean')
    
    # 3 groupby('chid')根據時間維度算norm_cnt(月平均消費次數(cnt取mean),再接一個normalize操作)
    card_cnt_col = df.columns[df.columns.str.contains('card')&df.columns.str.contains('cnt')]
    var3 = df.groupby('chid')[card_cnt_col].agg('mean')
    var3.loc[:,:] = normalize(var3)
    
    # 4 groupby('chid')根據時間維度算norm_cnt(月平均消費次數(cnt取mean),再接一個normalize操作)
    domestic_overseas_col = df.columns[(df.columns.str.contains('domestic')|df.columns.str.contains('overseas'))&(df.columns.str.contains('cnt'))]
    var4 = df.groupby('chid')[domestic_overseas_col].agg('mean')
    var4.loc[:,:] = normalize(var4)

    # 5 groupby('chid')根據時間維度算平均值(slam(正卡信用額度平均值),txn_amt(消費金額平均值),txn_cnt(消費次數平均值))
    other_mean_col = ['slam','txn_amt','txn_cnt']
    var5 = df.groupby('chid')[other_mean_col].agg('mean')

    # 1 2 3 4 5 concat 起來
    df_groupby_chid = pd.concat([var1,var2,var3,var4,var5],axis=1).reset_index()
    
    return df_groupby_chid

df_groupby_chid = create_groupby_chid_df(df)
assert df_groupby_chid['chid'].nunique() ==  500000 #確認應該有50萬筆

# 做資料預處理(one_hot和標準化轉換),把資料欄位都變成數值型態特徵
log.info("start one_hot and scale features")
from sklearn.preprocessing import scale
import warnings 
warnings.filterwarnings('ignore')

def preprocess_for_knn(df_groupby_chid):
    # 對類別欄位做one hot encoding
    categorical_df = pd.DataFrame()
    categorical_features = ['masts','educd','trdtp','naty','poscd','cuorg','gender_code','age','primary_card']
    for c_name in categorical_features:
        df_groupby_chid[c_name] = pd.to_numeric(df_groupby_chid[c_name])
    for c_name in categorical_features:
        one_hot = pd.get_dummies(df_groupby_chid[c_name])
        one_hot.columns = [ c_name + '_' + str(i) for i in one_hot.columns]
        if len(categorical_df)==0:
            categorical_df = one_hot
        else:
            categorical_df = pd.concat([categorical_df,one_hot],axis=1)
    df_groupby_chid = df_groupby_chid.drop(categorical_features,axis=1) # 刪掉原始類別欄位
    df_groupby_chid = pd.concat([df_groupby_chid,categorical_df],axis=1) # 加入one hot版本類別欄位
    
    # 做標準化轉換
    scale_col = set(df_groupby_chid.columns) - set(['chid'])
    scale_col = list(scale_col)
    df_groupby_chid[scale_col] = scale(df_groupby_chid[scale_col])
    
    return df_groupby_chid

df_groupby_chid_preprocessed = preprocess_for_knn(df_groupby_chid)

# 特徵維度太多先用pca進行降維
log.info('start excute PCA...')
from sklearn.decomposition import PCA
pca = PCA(n_components=args.pca_n)
X = df_groupby_chid_preprocessed.drop('chid',axis=1).values
X_pca = pca.fit_transform(X)

# 開始建立knn模型
log.info('start create knn model...')
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(
    n_neighbors=args.knn_k, 
    algorithm='ball_tree',
    leaf_size=args.knn_leaf_size,
    n_jobs=-1,
    p=args.knn_p).fit(X_pca)

# 保存
log.info('save model and data...')
import joblib
# 保存knn模型
joblib.dump(nbrs,'../model/nbrs.pkl')
# 保存X_pca
joblib.dump(X_pca,'../model/X_pca_for_knn.pkl')
# 保存df_groupby_chid_preprocessed
df_groupby_chid_preprocessed.to_feather('../data/df_groupby_chid_preprocessed.feather')
log.info('ALL DONE!')



