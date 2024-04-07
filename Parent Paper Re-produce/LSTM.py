import jieba
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import numpy as np
from keras import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Embedding, Dropout, Activation, Softmax
from datetime import datetime


# LSTM模型类
class Sentiment:
    def __init__(self, w2id, embedding_weights, Embedding_dim, maxlen, labels_category):
        self.Embedding_dim = Embedding_dim  # 嵌入层维度
        self.embedding_weights = embedding_weights  # 嵌入层权重
        self.vocab = w2id  # 词语的索引，从1开始编号
        self.labels_category = labels_category  # 分类的种类数
        self.maxlen = maxlen  # 输入维度
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        # input dim(140,100)
        model.add(Embedding(output_dim=self.Embedding_dim,
                            input_dim=len(self.vocab) + 1,
                            weights=[self.embedding_weights],
                            input_length=self.maxlen))
        model.add(Bidirectional(LSTM(50), merge_mode='concat'))  # 添加一个50节点的LSTM层
        model.add(Dropout(0.5))
        model.add(Dense(self.labels_category))  #
        model.add(Activation('softmax'))  # 添加一个softmax层
        model.compile(loss='categorical_crossentropy',  # 定义损失函数
                      optimizer='adam',  # 定义提升器
                      metrics=['accuracy'])  # 定义度娘指标
        model.summary()
        return model

    # 训练代码
    def train(self, X_train, y_train, n_epoch=5):
        a = datetime.now()
        # self.model.fit(X_train, y_train, batch_size=32, epochs=n_epoch,
        #                validation_data=(X_test, y_test))
        self.model.fit(X_train, y_train, batch_size=32, epochs=n_epoch, validation_split=0.1)
        b = datetime.now()
        print("模型训练时间:", (b - a).seconds, "秒")
        self.model.save('sentiment.h5')

    # 预测代码
    def predict(self, model_path, new_sen):
        model = self.model
        model.load_weights(model_path)
        new_sen_list = jieba.lcut(new_sen)
        sen2id = [self.vocab.get(word, 0) for word in new_sen_list]
        sen_input = pad_sequences([sen2id], maxlen=self.maxlen)
        res = model.predict(sen_input)[0]
        confidence = res.tolist()[np.argmax(res)]
        return np.argmax(res), confidence
