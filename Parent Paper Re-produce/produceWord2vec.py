import jieba
import time
from gensim.models.word2vec import Word2Vec
from Corpuspreprocess import corpusprocess

# 对中文文本进行预处理和切词操作

stopword_list = [k.strip() for k in open("stopwords.txt", encoding='utf-8') if k.strip() != '']  # 停用词


def train_word2vec(sentences, save_path):
    sentences_seg = []
    sen_str = "\n".join(sentences)
    # print("sen_str:",sen_str)
    res = jieba.lcut(sen_str)
    seg_str = " ".join(res)
    sen_list = seg_str.split("\n")
    for i in sen_list:
        sentences_seg.append(i.split())
    cutWords_sentences_seg = []
    for sen in sentences_seg:
        cutWords = [k for k in sen if k not in stopword_list]  # 去停用词
        cutWords_sentences_seg.append(cutWords)
    sentences_seg = cutWords_sentences_seg
    print("开始训练词向量")
    startTime = time.time()
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Word2Vec(sentences_seg,
                     size=100,  # 词向量维度
                     min_count=5,  # 词频阈值
                     window=5)  # 窗口大小
    model.save(save_path)
    print("一共花费%.2f秒" % (time.time() - startTime))
    return model


# 获取评论列表，sentences
sentences, _ = corpusprocess("train_flag.xlsx")
#
# 根据sentence是生成word2vec模型
model = train_word2vec(sentences, 'word2vec.model')
