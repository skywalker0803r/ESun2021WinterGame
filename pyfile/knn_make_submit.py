from tqdm.auto import tqdm
import pandas as pd
import random
import os
import numpy as np
import joblib
import logging
import coloredlogs
import time
import gc
import sys
import argparse
import itertools
from collections import Counter

# 顯示設定
gc.collect()
tqdm.pandas()
log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

# 超參數設定
parser = argparse.ArgumentParser()
parser.add_argument("name", help="1.exia 2.dynames 3.kyrios 4.virtue",type=str)
args = parser.parse_args()
def set_seed(seed = 42):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed
seed = set_seed(seed = 42)
debug_mode = True
start_dt = 12
log.info(f'debug_mode:{debug_mode}')

# 載入資料和模型
官方指認欄位 = ['2','6','10','12','13','15','18','19','21','22','25','26','36','37','39','48']
nbrs = joblib.load('../model/nbrs.pkl')
X_pca = joblib.load('../model/X_pca_for_knn.pkl')
df_groupby_chid_preprocessed = pd.read_feather('../data/df_groupby_chid_preprocessed.feather')
df = pd.read_feather('../data/2021玉山人工智慧公開挑戰賽冬季賽訓練資料集.feather')
df = df.loc[df.dt >= start_dt] # 取近期資料(太久的資料可能參考價值不高)
test_data = pd.read_feather('../data/需預測的顧客名單及提交檔案範例.feather')

# 因為要平行化所以切分資料成四份"1.exia 2.dynames 3.kyrios 4.virtue"
sp1,sp2,sp3 = int(len(test_data)/4),int(len(test_data)/4)*2,int(len(test_data)/4)*3
if args.name == 'exia':
    test_data = test_data.iloc[0:sp1,:] # exia掌管前面25%
elif args.name == 'dynames':
    test_data = test_data.iloc[sp1:sp2,:] #dynames掌管25%到50%
elif args.name == 'kyrios':
    test_data = test_data.iloc[sp2:sp3,:] #kyrios掌管50%到75%
elif args.name == 'virtue':
    test_data = test_data.iloc[sp3:,:] #virtue掌管75%到100%
else:
    log.info('args.name error!!!')
    sys.exit()
log.info(f'args.name:{args.name}')

# 一些函數都放在這裡
def chid2answer(chid,method='median'):
    if method in ['sum','mean','median']:
        a = df.loc[df.chid==chid,['shop_tag','txn_amt']].groupby('shop_tag').agg(method).sort_values(by='txn_amt',ascending=False)
    elif method in 'value_counts':
        a = df.loc[df.chid==chid,'shop_tag'].value_counts().to_frame()
    else:
        raise 'error'
    a['在指認欄位'] = False
    a.loc[list(set(a.index)&set(官方指認欄位)),'在指認欄位'] = True #有交集的部份做個記號
    answer = a[a['在指認欄位']==True].head(3)
    if len(answer) != 0:
        return answer.index.tolist()
    else:
        return []

def predict_function(chid):
    answer = chid2answer(chid) # 根據這個chid找答案但是不一定可以找到3個
    if len(answer) == 3:
        return answer
    else: # 若找不到三個
        idx = df_groupby_chid_preprocessed.loc[df_groupby_chid_preprocessed.chid==chid].index[0] #根據chid取得idx 
        distances, indices = nbrs.kneighbors(X_pca[[idx]]) # 根據idx取得PCA特徵
        chid_list = df_groupby_chid_preprocessed.loc[indices[0][-(nbrs.n_neighbors-1):]]['chid'].values.tolist() # 根據PCA特徵找到鄰居
        for nb_chid in chid_list: #對K個鄰居做遍歷
            nb_answer = chid2answer(nb_chid) # 鄰居的答案
            answer.extend(list(filter(lambda a: a not in answer, nb_answer))) #用鄰居答案對answer做擴充
            if len(answer) >= 3: # 如果補齊三個 return
                return answer[:3]
        remain = 3-len(answer) # 否則算還缺多少
        for _ in range(remain):
            answer.append(np.random.choice(list(set(官方指認欄位)-set(answer))))# 從官方指認欄位隨便補
        return answer

if debug_mode == True:
    submit = test_data.copy().head(42) # debug用少數樣本測試就好
if debug_mode == False:
    submit = test_data.copy() # 認真模式用全部

log.info('start predict...')
answer_list = submit['chid'].progress_apply(predict_function)
answer_list = answer_list.to_frame()['chid']
for i in [0,1,2]:
    submit[f'top{i+1}'] = answer_list.apply(lambda x:x[i]).values
save_path = f'submit_most_count_method_knn_{args.name}_{str(int(time.time()))}.csv' #保存路徑
submit.to_csv(save_path,index=False) #保存檔案
log.info(f'submission.shape:{submit.shape}')
log.info(f'submission.path:{save_path}')
log.info('ALL DONE!')