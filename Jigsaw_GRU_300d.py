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
        super(FastText, self).__init__()

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

        # x = x.mean(1).squeeze(1)
        x = F.relu(self.bn1(self.fc1(x)))
        logit = self.fc2(x)
        return logit


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print("EarlyStopping counter: {} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print("Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(self.val_loss_min, val_loss))
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

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
    # print("X_train[0:2] {}".format(X_train[0:2]))
    # print("y_train[0:2] {}".format(y_train[0:2]))
    # print("X_test[0:2] {}".format(X_test[0:2]))

    del tokenizer
    gc.collect()

    embeddings_index = load_embeddings(embed_path)
    embedding_matrix = build_matrix(word_index, embeddings_index)

    del embeddings_index
    gc.collect()

    return X_train, y_train, X_test, embedding_matrix

def train(X_train, y_train, X_test, embedding_matrix, batch_size=2048, embed_size=300, max_features=100000, n_splits=5):
    ##############            Training                         
    splits = list(KFold(n_splits).split(X_train, y_train))

    BATCH_SIZE = batch_size
    NUM_EPOCHS = 5
    
    train_preds = np.zeros((len(X_train)))
    test_preds = np.zeros((len(X_test)))

    x_test_cuda = torch.tensor(X_test, dtype=torch.float32).cuda()
    test = torch.utils.data.TensorDataset(x_test_cuda)    #pytorch自带的dataset
    test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

    for i, (train_idx, valid_idx) in enumerate(splits):
        x_train_fold = torch.tensor(X_train[train_idx], dtype=torch.float32).cuda()
        y_train_fold = torch.tensor(y_train[train_idx, np.newaxis], dtype=torch.float32).cuda()
        x_val_fold = torch.tensor(X_train[valid_idx], dtype=torch.float32).cuda()
        y_val_fold = torch.tensor(y_train[valid_idx, np.newaxis], dtype=torch.float32).cuda()
    
        model = RNN_GRU(embedding_matrix, max_features, embed_size)
        model.cuda()
    
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
        optimizer = torch.optim.Adam(model.parameters())
    
        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)
    
        train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False)
    
        early_stopping = EarlyStopping(patience=3, verbose=True)
    
        print("Fold {}".format(i + 1))
    
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
        
            model.eval()
            valid_preds_fold = np.zeros((x_val_fold.size(0)))
            test_preds_fold = np.zeros(len(X_test))
            avg_val_loss = 0.
            for i, (x_batch, y_batch) in enumerate(valid_loader):
                x_batch.float()
                y_batch.float()
                y_pred = model(x_batch).detach()
                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds_fold[i * BATCH_SIZE:(i+1) * BATCH_SIZE] = sigmoid(y_pred.cpu().numpy())[:, 0]
        
            elapsed_time = time.time() - start_time 
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, NUM_EPOCHS, avg_loss, avg_val_loss, elapsed_time))
        
            early_stopping(avg_val_loss, model)
        
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load('checkpoint.pt'))
    
        for i, (x_batch,) in enumerate(test_loader):
            y_pred = model(x_batch).detach()

            test_preds_fold[i * BATCH_SIZE:(i+1) * BATCH_SIZE] = sigmoid(y_pred.cpu().numpy())[:, 0]

        train_preds[valid_idx] = valid_preds_fold
        test_preds += test_preds_fold / len(splits)

###############                Submission              

    roc_auc_score(y_train>0.5, train_preds)

    submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')
    submission['prediction'] = test_preds
    submission.reset_index(drop=False, inplace=True)
    submission.head()

    submission.to_csv('submission.csv', index=False)

embed_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
train_path = '../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv'
test_path = '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'

X_train, y_train, X_test, embedding_matrix = load_data(embed_path, train_path, test_path)

train(X_train, y_train, X_test, embedding_matrix, batch_size=2048)
