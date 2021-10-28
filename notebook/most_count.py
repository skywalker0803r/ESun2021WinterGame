import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv('../data/2021玉山人工智慧公開挑戰賽冬季賽訓練資料集.gz.gz', compression='gzip')
df.columns = ['dt'] + df.columns[1:].tolist() # 更改欄位名稱
test_data = pd.read_csv('..\data\需預測的顧客名單及提交檔案範例.csv')
官方指認欄位 = [2,6,10,12,13,15,18,19,21,22,25,26,36,37,39,48]
官方指認欄位 = [str(i) for i in 官方指認欄位]
predicts = []
for chid in tqdm(test_data['chid'].values[:]):
    a = df.loc[df['chid']==chid,'shop_tag'].value_counts().to_frame()
    a['在指認欄位'] = False
    if len(list(set(a.index)&set(官方指認欄位))) != 0:
        a.loc[list(set(a.index)&set(官方指認欄位)),'在指認欄位'] = True
    answer = a[a['在指認欄位']==True].index.tolist()[:3]
    if len(answer) == 3:
        predicts.append(answer)
    else:
        for _ in range(3-len(answer)):
            answer.append(np.random.choice(官方指認欄位))
        predicts.append(answer)
predicts = np.array(predicts)
print(predicts.shape)
test_data.iloc[:,-3:] = predicts
print(test_data.shape)
test_data.to_csv('submit_most_count_method.csv')