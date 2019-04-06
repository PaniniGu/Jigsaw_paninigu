# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
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

def preprocess(data):
    '''
    Credit goes to 
    '''
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
    '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
    '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
    '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
    '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
    def clean_special_chars(text, puncts):
        for p in puncts:
            text = text.replace(p, ' ')
        return text
    data = data.astype(str).apply(lambda x: clean_special_chars(x, puncts))

    contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", \
                            "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not",\
                            "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", \
                            "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you",\
                            "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", \
                            "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", \
                            "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", \
                            "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",\
                            "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",\
                            "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", \
                            "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",\
                            "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", \
                            "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", \
                            "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", \
                            "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", \
                            "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", \
                            "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is",\
                            "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", \
                            "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", \
                            "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", \
                            "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  \
                            "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", \
                            "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", \
                            "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", \
                            "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", \
                            "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have",\
                            "y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", \
                            "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
    def clean_contractions(text, mapping):
        specials = ["’", "‘", "´", "`"]
        for s in specials:
            text = text.replace(s, "'")
        text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
        return text

    data = data.astype(str).apply(lambda x: clean_contractions(x, contraction_mapping))
    
    return data



def load_data(embed_path, train_path, test_path, maxlen=220, embed_size=300, max_features=100000):
    # embed_path = '../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt'
    train = pd.read_csv(train_path)
    print('loaded %d train records' % len(train))
    test = pd.read_csv(test_path, index_col='id')
    print('loaded %d test records' % len(test))

    train['comment_text'] = preprocess(train['comment_text'])
    test['comment_text'] = preprocess(test['comment_text'])

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

embed_path = '../input/glove840b300dtxt/glove.840B.300d.txt'
train_path = '../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv'
test_path = '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'

X_train, y_train, X_test, embedding_matrix = load_data(embed_path, train_path, test_path)

train(X_train, y_train, X_test, embedding_matrix, batch_size=512)