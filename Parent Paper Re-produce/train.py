import torch as t
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.utils.data
from data_loader import MyData
from model import SLCABG
import data_util


device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
SENTENCE_LENGTH = 12
WORD_SIZE = 35000
EMBED_SIZE = 768

######################
from LSTM import Sentiment
from gensim.models.word2vec import Word2Vec
from dataPreprocess import generate_id2wec

model = Word2Vec.load('word2vec.model')

w2id,embedding_weights = generate_id2wec(model)
label_dic = {0:"Fake",1:"Real"}
label_dic2 = {0:"negative",1:"positive"}

senti = Sentiment(w2id,embedding_weights,100,200,2)

##################################

if __name__ == '__main__':
    # sentences, label, word_vectors = data_util.process_data(SENTENCE_LENGTH, WORD_SIZE, EMBED_SIZE)
    # x_train, x_test, y_train, y_test = train_test_split(sentences, label, test_size=0.2)


#######################################
    sentences, label, word_vectors = data_util.process_data(SENTENCE_LENGTH, WORD_SIZE, EMBED_SIZE)
    indices = list(range(100000))
    train_indices, test_indices,x_train, x_test, y_train, y_test = train_test_split(indices,sentences[1], label, random_state=42,test_size=0.2)
    train_sentences = [sentences[0][i] for i in train_indices]
    test_sentences = [sentences[0][i] for i in test_indices]
    
    train_data_loader = torch.utils.data.DataLoader(MyData(x_train, y_train), 32, True)
    test_data_loader = torch.utils.data.DataLoader(MyData(x_test, y_test), 32, False)


##########################################



    net = SLCABG(EMBED_SIZE, SENTENCE_LENGTH, word_vectors).to(device)
    optimizer = t.optim.Adam(net.parameters(), 0.01)
    criterion = nn.CrossEntropyLoss()
    tp = 1
    tn = 1
    fp = 1
    fn = 1
    for epoch in range(15):
        for i, (cls, sentences) in enumerate(train_data_loader):
            optimizer.zero_grad()
            sentences = sentences.type(t.LongTensor).to(device)
            cls = cls.type(t.LongTensor).to(device)
            out = net(sentences)
            _, predicted = torch.max(out.data, 1)
            predict = predicted.cpu().numpy().tolist()
            pred = cls.cpu().numpy().tolist()
            for f, n in zip(predict, pred):
                if f == 1 and n == 1:
                    tp += 1
                elif f == 1 and n == 0:
                    fp += 1
                elif f == 0 and n == 1:
                    fn += 1
                else:
                    tn += 1
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * r * p / (r + p)
            acc = (tp + tn) / (tp + tn + fp + fn)
            loss = criterion(out, cls).to(device)
            loss.backward()
            optimizer.step()
            if (i + 1) % 1 == 0:
                print("epoch:", epoch + 1, "step:", i + 1, "loss:", loss.item())
                print('acc', acc, 'p', p, 'r', r, 'f1', f1)
##############################################################
    torch.save(net, 'model.pth')
##############################################################

#############################################################
    net = torch.load('model.pth')
    net.eval()
    print('==========================================================================================')
    with torch.no_grad():
        tp = 1
        tn = 1
        fp = 1
        fn = 1
        for i,(cls, sentences) in zip(test_sentences,test_data_loader):
###########################################   
            res = senti.predict("./sentiment.h5",i )  # predict if fake reviews
            index = res.index(max(res))  
            f_label = label_dic[index]  # prediction label index
            if index == 1: # if labeled real review
                confidence = res[1]  # confidence level，when predict real review，then it's the probablity of being real
                Fakedegrees = str(round(res[0] * 100, 2)) + "%"  # fake degress, if predict real, then it's the possibility of being fake review
            else: # if labeled fake
                confidence = res[0]  # confidence level, when predict fake, it's the possibility that review is fake
                Fakedegrees = str(round(res[0], 2) * 100) + "%"  # fake degrees，if predict fake，it's the possibility of being fake
            print("Comment:", i)
            print("If fake:", f_label)
            print("Confidence Level:", confidence)
            print("Fake Degrees:", Fakedegrees)
###########################################
            sentences = sentences.type(t.LongTensor).to(device)
            cls = cls.type(t.LongTensor).to(device)
            out = net(sentences)
            _, predicted = torch.max(out.data, 1)
            predict = predicted.cpu().numpy().tolist()
            pred = cls.cpu().numpy().tolist()
            for f, n in zip(predict, pred):
                if f == 1 and n == 1:
                    tp += 1
                elif f == 1 and n == 0:
                    fp += 1
                elif f == 0 and n == 1:
                    fn += 1
                else:
                    tn += 1
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * r * p / (r + p)
        acc = (tp + tn) / (tp + tn + fp + fn)
        print("Sentiment Analysis Model:",'acc', acc, 'p', p, 'r', r, 'f1', f1) 
        #it only measures the sentiment analysis part, will consider the fake review detection if have more ground truth data
