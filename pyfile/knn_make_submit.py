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
gc.collect()
tqdm.pandas()
log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

# 參數設定
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
df = df.loc[df.dt >= start_dt] # 取近期資料
test_data = pd.read_feather('../data/需預測的顧客名單及提交檔案範例.feather')

# 函數
def chid2answer(chid): #chid到answer的映射
    a = df.loc[df['chid']==chid,'shop_tag'].value_counts().to_frame()
    a['在指認欄位'] = False
    if len(list(set(a.index)&set(官方指認欄位))) != 0: #如果跟官方指定欄位有交集
        a.loc[list(set(a.index)&set(官方指認欄位)),'在指認欄位'] = True #有交集的部份做記號
    answer = a[a['在指認欄位']==True].index.tolist()[:3] # 取有交集的部份前三名返回
    return answer

def predict_function(chid): # 預測函數
    answer = chid2answer(chid) # 根據這個chid做預測
    if len(answer) == 3: # 如果成功找到三個直接return
        return answer
    else:
        remain = 3-len(answer) # 否則計算還缺多少
        idx = df_groupby_chid_preprocessed.loc[df_groupby_chid_preprocessed.chid==chid].index[0] # 根據chid找到該筆樣本的"idx"
        distances, indices = nbrs.kneighbors(X_pca[[idx]]) # 根據該樣本的idx找到該筆樣本的"PCA特徵"然後取得"鄰居的indices"
        chid_list = df_groupby_chid_preprocessed.loc[indices[0][-(nbrs.n_neighbors-1):]]['chid'].values.tolist() # 根據"鄰居的indices"取得"chid_list(鄰居們)"
        answer_list = [chid2answer(chid) for chid in chid_list]# 根據"chid_list"取得"answer_list"
        # 將answer_list做一維展開========
        answer_list_ = []
        for i in answer_list:
            answer_list_.extend(i)
        answer_list = answer_list_
        # 將answer_list做一維展開========
        answer_list = list(set(answer_list) - set(answer)) # 如果"answer"裡面已經有了則"去除"
        for _ in range(remain):
            if len(answer_list) != 0: #如果answer_list不等於0
                shop_tag = answer_list[0] # 從answer_list選第一個shop_tag(選第一個是因為distances比較近)
                answer.append(shop_tag) # 加入shop_tag至answer
                answer_list.remove(shop_tag) # 用過的shop_tag記得移除
            else:
                shop_tag = np.random.choice(官方指認欄位)
                answer.append(shop_tag) # 加入shop_tag至answer
        return answer


log.info('start predict...')
answer_list = submit['chid'].progress_apply(predict_function)
answer_list = answer_list.to_frame()['chid']

log.info('create submit...')

if debug_mode == True:
    submit = test_data.copy().head(42)
if debug_mode == False:
    submit = test_data.copy()

for i in [0,1,2]:
    submit[f'top{i+1}'] = answer_list.apply(lambda x:x[i]).values

save_path = f'submit_most_count_method_knn_{str(int(time.time()))}.csv'
submit.to_csv(save_path,index=False)

log.info(f'submission.shape:{submission.shape}')
log.info(f'submission.path:{save_path}')
log.info('ALL DONE!')