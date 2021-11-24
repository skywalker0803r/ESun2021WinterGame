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

# 終端機顯示設定
gc.collect()
tqdm.pandas()
log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

# 是否啟用debug模式
debug_mode = True
log.info(f'debug_mode:{debug_mode}')

# 超參數設定
parser = argparse.ArgumentParser()
parser.add_argument("name", help="1.exia 2.dynames 3.kyrios 4.virtue",type=str)
args = parser.parse_args()
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
dotargs = dotdict({
    'knn_k':42,
    'predict_method':'most_common',
    'start_dt':12,
    'seed':42,
})
def set_seed(seed):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed
seed = set_seed(seed = dotargs.seed)
log.info(f'setting args is :{dotargs}')

# 載入資料和模型
官方指認欄位 = ['2','6','10','12','13','15','18','19','21','22','25','26','36','37','39','48'] #官方只認這些
nbrs = joblib.load('../model/nbrs.pkl') # knn模型
X_pca = joblib.load('../model/X_pca_for_knn.pkl') #PCA過後的特徵
df_groupby_chid_preprocessed = pd.read_feather('../data/df_groupby_chid_preprocessed.feather')#groupby_chid版本的訓練資料
df = pd.read_feather('../data/2021玉山人工智慧公開挑戰賽冬季賽訓練資料集.feather')#訓練資料
df = df.loc[df.dt >= dotargs.start_dt] # 取近期資料(太久的資料可能參考價值不高)
test_data = pd.read_feather('../data/需預測的顧客名單及提交檔案範例.feather')#用來製作submit的資料
if debug_mode == True:
    submit = test_data.copy().sample(42) # debug用少數樣本測試就好
if debug_mode == False:
    submit = test_data.copy() # 認真模式用全部

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
def check_answer(answer,n_class=3):
    assert type(answer) == type([]) #記得確認是list型別
    assert len(np.unique(answer)) == n_class #確認三個shop_tag不重複
    return answer #返回答案[top1,top2,top3]

def get_answer(chid_list):
    a = df.loc[df.chid.isin(chid_list),'shop_tag'].value_counts().to_frame() # 根據chid_list計算value_counts
    a['在指認欄位'] = False
    a.loc[list(set(a.index)&set(官方指認欄位)),'在指認欄位'] = True # 與官方認證的16個欄位取交集
    answer = a[a['在指認欄位']==True].head(3) # 取前三名
    if len(answer) != 0: # 如果答案不為空列表
        return answer.index.tolist() #返回list
    else:
        return [] #返回list

def get_k_chids(chid,k):
    idx = df_groupby_chid_preprocessed.loc[df_groupby_chid_preprocessed.chid==chid].index[0] #根據chid取得樣本的idx
    distances, indices = nbrs.kneighbors(X_pca[[idx]]) # 根據樣本的idx取得PCA特徵,和鄰居的indices
    chid_list = df_groupby_chid_preprocessed.loc[indices[0]]['chid'].values.tolist() # 根據鄰居的indices找到鄰居的chid
    return chid_list[:k] #只選擇最近的k個鄰居

def chid2answer(chid_list,k=2,max_k=dotargs.knn_k):
    if k >= max_k:# 如果達到max_k隨便選三個shop_tag然後return(max_k設定愈高,觸發機率愈低)
        answer = [] # 初始化空的 answer list
        for _ in range(3): # 補滿 top1 top2 top3
            answer.append(np.random.choice(list(set(官方指認欄位)-set(answer))))# 從官方指認欄位隨便補(三個不能重複)
        return check_answer(answer) # 返回前檢查
    else: # 否則根據chid_list去猜shop_tag
        answer = get_answer(chid_list)
        if len(answer) == 3:# 如果湊齊三個答案直接return
            return check_answer(answer)# 返回前檢查
        else: # 如果湊不齊則用chid_list[0]去找k個鄰居再遞迴運算
            chid_list = get_k_chids(chid_list[0],k) #k初始值是2,等於找自己跟一個鄰居
            return chid2answer(chid_list,k=k+1)

# 開始執行預測
log.info('start predict...')
answer_list = submit['chid'].progress_apply(lambda chid:chid2answer([chid])).to_frame()['chid']
# 用預測完的answer_list對submit的top1,top2,top3做補上的動作
for i in [0,1,2]:
    submit[f'top{i+1}'] = answer_list.apply(lambda x:x[i]).values
# 保存
save_path = f'submit_most_count_method_knn_{args.name}_{str(int(time.time()))}.csv' #保存路徑
submit.to_csv(save_path,index=False) # 執行保存檔案的動作
log.info(f'submission.shape:{submit.shape}')
log.info(f'submission.path:{save_path}')
log.info('ALL DONE!')
