import pandas as pd
import numpy as np
import time
import os
import random
from tqdm import tqdm
from apyori import apriori
import joblib
import logging
import coloredlogs
import gc
gc.collect()
log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

# setting args
def set_seed(seed = 42):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed
seed = set_seed(seed = 42)
debug_mode = False
start_dt = int(24-12)
association_rules_n = 5290
官方指認欄位 = ['2','6','10','12','13','15','18','19','21','22','25','26','36','37','39','48']
train_path = '../data/2021玉山人工智慧公開挑戰賽冬季賽訓練資料集.feather'
test_path = '../data/需預測的顧客名單及提交檔案範例.feather'

log.info('load data...')
df = pd.read_feather(train_path)
test_data = pd.read_feather(test_path)
log.info('load data done!')

log.info('crate shop_tag_mapping...')
data = []
for chid in tqdm(df.sample(association_rules_n,random_state=seed)['chid'].values):
    data.append(df.loc[(df.chid==chid)&(df.dt>=start_dt),'shop_tag'].value_counts().index.tolist())
association_rules = apriori(
    data, 
    min_support=0.1, 
    min_confidence=0.1, 
    min_lift=1.1,
    max_length=2,
    ) 
association_results = list(association_rules)
mapping = {}
for key in 官方指認欄位:
    mapping[key] = []
for item in association_results:
    pair = item[0] 
    items = [x for x in pair]
    source = items[0]
    target = items[1]
    if (source in 官方指認欄位) and (target in 官方指認欄位):
        mapping[source].append(target)
log.info('crate shop_tag_mapping done!')

log.info('start predict...')
if debug_mode == True:
    n = 100
    chid_index = test_data.head(n)['chid'].values
    log.info(f'use debug_mode n = {n}')
if debug_mode == False:
    n = len(test_data)
    chid_index = test_data['chid'].values
    log.info(f'use real_mode n = {n}')

predicts = []
for chid in tqdm(chid_index):
    a = df.loc[(df['chid']==chid)&(df['dt']>=start_dt),'shop_tag'].value_counts().to_frame() # 根據chid對shop_tag做value_counts()
    a['在指認欄位'] = False
    if len(list(set(a.index)&set(官方指認欄位))) != 0: # 判斷這個chid出現過的shop_tag跟官方指認欄位是否有交集
        a.loc[list(set(a.index)&set(官方指認欄位)),'在指認欄位'] = True
    answer = a[a['在指認欄位']==True].index.tolist()[:3] # 取value_counts()前三大的當作答案
    if len(answer) == 3:# 如果成功取出前三大的predicts直接指派answer
        predicts.append(answer)
    else: # 如果沒有找到三個的話
        for _ in range(3-len(answer)):
            if len(answer) == 0: # 如果找到0個 則隨機指派
                answer.append(np.random.choice(官方指認欄位))
            else: # 否則建立候選清單
                tag_list = mapping[np.random.choice(answer)]
                if len(tag_list) > 0: # 如果候選清單大於0
                    answer.append(np.random.choice(tag_list)) # 從候選清單挑
                else: # 否則從官方指認欄位挑
                    answer.append(np.random.choice(官方指認欄位))
        predicts.append(answer)
log.info('predict done!')
log.info('create submit...')
submission = test_data.copy()
submission.iloc[:n,-3:] = np.array(predicts)
save_path = f'submit_most_count_method_apriori_{str(int(time.time()))}.csv'
submission.to_csv(save_path,index=False)
log.info('create submit done!')
log.info(f'submission.shape:{submission.shape}')
log.info('ALL DONE!')