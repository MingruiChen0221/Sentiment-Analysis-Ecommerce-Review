import pandas as pd
from sklearn.utils import shuffle


# 该模块是给原始语料做预处理过程，并返会sentence和labels两个list
# sentence为网评内容，labels为sentence每一条评论对应的标签，其中1代表好评，0代表差评


def make_label(star):
    if star == 'F':
        return 1
    else:
        return 0


# 处理原始语料，并返回评论列表sentence,和标签列表labels
def corpusprocess(filepath):
    # data = pd.read_csv(filepath, engine='python')
    data = pd.read_excel(filepath)
    data = shuffle(data)
    data = data.reset_index(drop=True)
    print(data['star'].value_counts().sort_index())
    data["sentiment"] = data.star.apply(make_label)  # 将原始标签生成0，1标签，并保存到data["sentiment"]
    sentences = data["reviewbody"].astype(str)
    labels = data["sentiment"]
    return sentences, labels

# sentences,labels = corpusprocess("train.csv")
