from LSTM import Sentiment
from gensim.models.word2vec import Word2Vec
from dataPreprocess import generate_id2wec

model = Word2Vec.load('word2vec.model')

w2id,embedding_weights = generate_id2wec(model)
label_dic = {0:"假评",1:"真评"}


#设置模型参数
senti = Sentiment(w2id,embedding_weights,100,200,2)



#在sen_new_list列表中替换你想测试的评论，可以是一条和多条
sen_new_list = [
    "特别特别满意，超级好看，客服很有礼貌，很客气，良心商家，支持！?质量很好，买的很合我心意?，棒棒哒！?需要的亲们可以放心购买真的推荐啊，快递速度特别的快，爱了爱了宝贝收到了，质量很不错，包装精致，材质优秀，比想象中好，送礼自用非常合适，下次继续购买产品包装精致美观大气，目前用着非常不错，真实评价，希望可以帮到集美们"
    "棒棒哒！?需要的亲们可以放心购买真的推荐啊",
    "目前用着非常不错，真实评价"
    ]

#预测代码
#加载模型并进行预测
# print("输入了",len(sen_new_list),"条评论")
# for i in  range(len(sen_new_list)):
#     sen_new = sen_new_list[i]
#     pre, confidence = senti.predict("./sentiment.h5", sen_new)
#     print("评论",i)
#     print("评价内容:", sen_new)
#     print("评价判断:", label_dic[pre])
#     print("置信度（可信度）:", confidence)



while 1:
    sen_new_list = input('''请输入你要检测的评论(一个或多个，多个评论时评论用#号隔开):\n''').split("#")

    #sen_new = input()
    for sen_new in sen_new_list:
        res = senti.predict("./sentiment.h5", sen_new)  # 预测结果为输入句子是假和真两种评论的概率，保存到res中
        index = res.index(max(res))  # 返回概率数组res中最大值的下标
        label = label_dic[index]  # 预测的结果标签
        if index == 1: #此时label标签值为真
            confidence = res[1]  # 置信度，此时评价预测为真，则其为真的概率就为是置信度
            Fakedegrees = str(round(res[0] * 100, 2)) + "%"  # 虚假程度，此时评价预测为真，则其为假的概率就为是虚假度
        else: #此时label标签值为假
            confidence = res[0]  # 置信度，此评论为假时，其为假的概率就是置信度
            Fakedegrees = str(round(res[0], 2) * 100) + "%"  # 虚假程度，此评论为假时，其为假的概率就是虚假度
        print("评论：", sen_new)
        print("预测结果：", label)
        print("置信度：", confidence)
        print("虚假度：", Fakedegrees)
