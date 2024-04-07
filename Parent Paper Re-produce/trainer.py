import pandas as pd
import logging
import time
from produceWord2vec import train_word2vec
from gensim.models.word2vec import Word2Vec

from LSTM import Sentiment
from dataPreprocess import generate_id2wec, prepare_data

from Corpuspreprocess import corpusprocess

# sentences, labels = corpusprocess("train.csv")
sentences, labels = corpusprocess("./train_flag.xlsx")



print(len(sentences))

# 获取已经生成的word2vec词向量模型
model = Word2Vec.load('word2vec.model')

# 获取词语的索引的w2id,和嵌入权重embedding_weights
w2id, embedding_weights = generate_id2wec(model)

# 数据预处理
# x_train,训练集词向量
# y_trian,训练集标签
# x_test，测试集词向量
# y_test 测试集标签
x_train, y_trian, x_test, y_test = prepare_data(w2id, sentences, labels, 200)

# 设置模型参数
senti = Sentiment(w2id, embedding_weights, 100, 200, 2)

# 训练，自动保存为sentiment.h5
# senti.train(x_train, y_trian, x_val, y_val, 60)
senti.train(x_train, y_trian, 60)

# 利用测试集验证模型效果
y_pre = senti.model.predict(x_test)


y_test = y_test.argmax(axis=1)
y_pre = y_pre.argmax(axis=1)

from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, y_pre, digits=4))

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings("ignore")
conmatrix = confusion_matrix(y_test, y_pre)


df_cm=pd.DataFrame(conmatrix, index=['T', 'F'], columns=['T', 'F'])

plt.figure(figsize=(4,3))
plt.title('conmatrix')
sns.heatmap(df_cm,annot=True,cmap="BuPu",fmt='.20g')
plt.show()