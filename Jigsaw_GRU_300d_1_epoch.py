# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#
import numpy as np 
import pandas as pd 
import os
import gc 
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data

tqdm.pandas()

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        n, t = x.size(0), x.size(1) 
        # merge batch and seq dimensions
        x_reshape = x.contiguous().view(t * n, x.size(2))
        y = self.module(x_reshape)
        # we have to reshape Y
        y = y.contiguous().view(n, t, y.size()[1])
        return y

class RNN_GRU(nn.Module):
    """docstring for FastText"""
    def __init__(self, emdedding_matrix, max_features, embed_size):
        super(RNN_GRU, self).__init__()

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        D = 300
        self.tdfc1 = nn.Linear(D, 256)
        self.td1 = TimeDistributed(self.tdfc1)
        self.tdbn1 = nn.BatchNorm2d(1)

        self.hidden_num = hidden_num = 256
        self.rnn1 = nn.GRU(256, hidden_num, bidirectional=True, batch_first=True)

        C = 1
        self.fc1 = nn.Linear(hidden_num*2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, C)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding(x.long())
        x.detach()
        x = F.relu(self.tdbn1(self.td1(x).unsqueeze(1))).squeeze(1)

        h0_1 = Variable(torch.randn(2, batch_size, self.hidden_num))
        h0_1 = h0_1.cuda()
        _, z1 = self.rnn1(x, h0_1)
        x = z1[-2:].transpose(0, 1).contiguous().view(batch_size, -1)

#         x = x.mean(1).squeeze(1)
        x = F.relu(self.bn1(self.fc1(x)))
        logit = self.fc2(x)
        return logit

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

def load_embeddings(embed_dir):
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in tqdm(open(embed_dir)))
    return embedding_index

def build_embedding_matrix(word_index, embeddings_index, max_features, lower=True, verbose=True):
    embedding_matrix = np.zeros((max_features, 300))
    for word, i in tqdm(word_index.items(), disable = not verbose):
        if lower:
            word = word.lower()
        if i >= max_features:continue
        try:
            embedding_vector = embeddings_index[word]
        except:
            embedding_vector = embeddings_index["unknown"]

        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def build_matrix(word_index, embeddings_index):
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embeddings_index[word]
        except:
            embedding_matrix[i] = embeddings_index["unknown"]
    return embedding_matrix


def load_data(embed_path, train_path, test_path, maxlen=220, embed_size=300, max_features=100000):
    # embed_path = '../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt'
    train = pd.read_csv(train_path)
    print('loaded %d train records' % len(train))
    test = pd.read_csv(test_path, index_col='id')
    print('loaded %d test records' % len(test))

    # maxlen = 220
    # max_features = 100000
    # embed_size = 300
    tokenizer = Tokenizer(num_words=max_features, lower=True)
    print('fitting tokenizer')
    tokenizer.fit_on_texts(list(train['comment_text']) + list(test['comment_text'])) #使用一系列文档来生成token词典，参数为list类，每个元素为一个文档。
    word_index = tokenizer.word_index    #一个dict，保存所有word对应的编号id，从1开始
    X_train = tokenizer.texts_to_sequences(list(train['comment_text']))  #将多个文档转换为word下标的向量形式,shape为[len(texts)，len(text)](文档数，每条文档的长度)
    train['target'] = train['target'].apply(lambda x: 1 if x > 0.5 else 0)
    y_train = train['target'].values
    X_test = tokenizer.texts_to_sequences(list(test['comment_text']))

    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)

    del tokenizer
    gc.collect()

    embeddings_index = load_embeddings(embed_path)
    embedding_matrix = build_matrix(word_index, embeddings_index)

    del embeddings_index
    gc.collect()

    return X_train, y_train, X_test, embedding_matrix

def train(X_train, y_train, X_test, embedding_matrix, batch_size=512, embed_size=300, max_features=100000):
    ##############            Training
    BATCH_SIZE = batch_size
    NUM_EPOCHS = 1
    
    # train_preds = np.zeros((len(X_train)))
    test_preds = np.zeros((len(X_test)))

    x_test_cuda = torch.tensor(X_test, dtype=torch.float32).cuda()
    test = torch.utils.data.TensorDataset(x_test_cuda)    #pytorch自带的dataset
    test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

    model = RNN_GRU(embedding_matrix, max_features, embed_size)
    model.cuda()
    
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters())
    
    X_train = torch.tensor(X_train, dtype=torch.float32).cuda()
    y_train = torch.tensor(y_train, dtype=torch.float32).cuda()
    train = torch.utils.data.TensorDataset(X_train, y_train)
    # valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    # valid_loader = torch.utils.data.DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False)
    
        # early_stopping = EarlyStopping(patience=3, verbose=True)
    
        # print("Fold {}".format(i + 1))
    print(model)
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        model.train()
        avg_loss = 0.
        for x_batch, y_batch in tqdm(train_loader, disable=True):
            optimizer.zero_grad()
            x_batch.float()
            y_batch.float()
#                 print("x_batch.size():{}".format(x_batch.size()))
#                 print("y_batch.size():{}".format(y_batch.size()))
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        
        print('Epoch {}/{} \t loss={:.4f}'.format(
                epoch + 1, NUM_EPOCHS, avg_loss))
        
        # load the last checkpoint with the best model
    # model.load_state_dict(torch.load('checkpoint.pt'))
    model.eval()
    for i, (x_batch,) in enumerate(test_loader):
        y_pred = model(x_batch).detach()

        test_preds[i * BATCH_SIZE:(i+1) * BATCH_SIZE] = sigmoid(y_pred.cpu().numpy())[:, 0]

###############                Submission              

    # roc_auc_score(y_train>0.5, train_preds)

    submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')
    submission['prediction'] = test_preds
    submission.reset_index(drop=False, inplace=True)
    submission.head()

    submission.to_csv('submission.csv', index=False)

embed_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
train_path = '../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv'
test_path = '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'

X_train, y_train, X_test, embedding_matrix = load_data(embed_path, train_path, test_path)

train(X_train, y_train, X_test, embedding_matrix, batch_size=512)
