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

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
dotargs = dotdict({
    'knn_k':10,
    'predict_method':'most_common',
    'seed':43,
})
log.info(f'setting args is :{dotargs}')

def set_seed(seed):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed
seed = set_seed(seed = dotargs.seed)
debug_mode = False
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
def chid2answer(chid,method='value_counts'):
    if method in ['sum','mean','median']:
        a = df.loc[df.chid==chid,['shop_tag','txn_amt']].groupby('shop_tag').agg(method).sort_values(by='txn_amt',ascending=False)
    elif method in ['value_counts']:
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

def predict_function_distance_first(chid):
    answer = chid2answer(chid) # 根據這個chid找答案但是不一定可以找到3個
    if len(answer) == 3:
        return answer
    else: # 若找不到三個
        idx = df_groupby_chid_preprocessed.loc[df_groupby_chid_preprocessed.chid==chid].index[0] #根據chid取得idx 
        distances, indices = nbrs.kneighbors(X_pca[[idx]]) # 根據idx取得PCA特徵
        chid_list = df_groupby_chid_preprocessed.loc[indices[0][-(nbrs.n_neighbors-1):]]['chid'].values.tolist() # 根據PCA特徵找到鄰居
        for nb_chid in chid_list[:dotargs.knn_k]: #對K個鄰居做遍歷
            nb_answer = chid2answer(nb_chid) # 鄰居的答案
            answer.extend(list(filter(lambda a: a not in answer, nb_answer))) #用鄰居答案對answer做擴充
            if len(answer) >= 3: # 如果補齊三個 return
                return answer[:3]
        remain = 3-len(answer) # 否則算還缺多少
        for _ in range(remain):
            answer.append(np.random.choice(list(set(官方指認欄位)-set(answer))))# 從官方指認欄位隨便補
        return answer

def predict_function_most_common(chid): # 預測函數
    answer = chid2answer(chid) # 根據這個chid做預測
    if len(answer) == 3: # 如果成功找到三個直接return
        assert type(answer) == type([]) #記得確認是list型別
        assert len(np.unique(answer)) == 3 #確認三個shop_tag不重複
        return answer 
    else:
        remain = 3-len(answer) # 否則計算離三個答案還缺多少
        idx = df_groupby_chid_preprocessed.loc[df_groupby_chid_preprocessed.chid==chid].index[0] # 根據chid找到該筆樣本的"idx"
        distances, indices = nbrs.kneighbors(X_pca[[idx]]) # 根據該樣本的"idx"找到該筆樣本的"PCA特徵"進而取得"鄰居的indices"(其中距離近的indices自動排前面)
        chid_list = df_groupby_chid_preprocessed.loc[indices[0][-(nbrs.n_neighbors-1):]]['chid'].values.tolist() # 根據"鄰居的indices"取得"chid_list(鄰居們)"
        chid_list = chid_list[:dotargs.knn_k]
        answer_list = [chid2answer(chid) for chid in chid_list] # 根據"chid_list"取得"answer_list"
        answer_list = list(itertools.chain(*answer_list)) # 將answer_list做"一維展開"
        answer_list = list(filter(lambda a: a not in answer, answer_list)) # 如果該shop_tag在"answer"裡面已經有了則從answer_list"去除"
        for _ in range(remain): # 遍歷剩餘數量做補上的動作,直到補滿3個答案
            if len(answer_list) != 0: #如果answer_list不等於0
                shop_tag = Counter(answer_list).most_common()[0][0] # 從answer_list選most_common的shop_tag(出現頻率比較多可能是正確答案)
                answer.append(shop_tag) # 加入該shop_tag至answer
                answer_list = list(filter(lambda a: a not in answer, answer_list)) # 記得把answer有的shop_tag從answer_list做刪除
            else: #如果answer_list等於0
                answer_list = 官方指認欄位 # 既然answer_list等於0估解將官方指認欄位當作answer_list
                shop_tag = np.random.choice(list(set(answer_list)-set(answer))) # 隨機選但是answer裡面已經有的就不要選,官方規定的
                answer.append(shop_tag) # 加入shop_tag至answer
                answer_list = list(filter(lambda a: a not in answer, answer_list)) # 記得把answer有的shop_tag從answer_list做刪除
        assert type(answer) == type([]) #確認是list型別
        assert len(np.unique(answer)) == 3 #確認三個shop_tag不重複
        return answer # 返回答案(類型list)

if debug_mode == True:
    submit = test_data.copy().head(42) # debug用少數樣本測試就好
if debug_mode == False:
    submit = test_data.copy() # 認真模式用全部

if dotargs.predict_method == 'most_common':
    predict_function = predict_function_most_common
if dotargs.predict_method == 'distance_first':
    predict_function = predict_function_distance_first

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
