# This program applies GRU to POS tagging.

import collections
import nltk
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Dense, SpatialDropout1D, GRU, RepeatVector, Activation, LSTM, Input
import numpy as np
from keras.layers.wrappers import TimeDistributed

fsents = open('./treebank_sents.txt','w')
fposs = open('./treebank_poss.txt','w')

sents = nltk.corpus.treebank.tagged_sents()
for sent in sents:
    words, poss = [], []
    for word, pos in sent:
        if pos == '-NONE-':
            continue;
        words.append(word)
        poss.append(pos)
    fsents.write("{:s}\n".format(" ".join(words)))
    fposs.write("{:s}\n".format(" ".join(poss)))
fsents.close()
fposs.close()

def parse_sentence(filename):
    word_freqs = collections.Counter()
    maxlen = 0
    num_recs = 0
    with open(filename,'r') as fin:
        for line in fin:
            words = line.strip().lower().split()
            maxlen = max(maxlen, len(words))
            for word in words:
                word_freqs[word] += 1
            num_recs += 1
    return word_freqs, num_recs, maxlen

s_wordfreqs, num_recs, maxlen = parse_sentence('./treebank_sents.txt')
t_wordfreqs, num_recs, maxlen = parse_sentence('./treebank_poss.txt')

MAX_SEQLEN = 250
S_MAX_FEATURES = 10947
T_MAX_FEATURES = 45

s_vocabsize = min(len(s_wordfreqs), S_MAX_FEATURES) + 1
s_word2index = {x[0]: i+1 for i, x in enumerate(s_wordfreqs.most_common(S_MAX_FEATURES))}
s_word2index['PAD'] = 0
#s_word2index['UNK'] = 1
s_index2word  = {v:k for k, v in s_word2index.items()}

t_vocabsize = len(t_wordfreqs) + 1
t_word2index = {x[0]: i+1 for i, x in enumerate(t_wordfreqs.most_common(T_MAX_FEATURES))}
t_word2index['PAD'] = 0
t_index2word = {v:k for k, v in t_word2index.items()}

def build_tensor(filename, numrecs, word2index, maxlen, make_categorical = False, num_classes = 0):
    data = np.empty((numrecs,),dtype=list)
    with open(filename, 'r') as fin:
        for i, line in enumerate(fin):
            wids = []
            for word in line.strip().lower().split():
                if word in word2index:
                    wids.append(word2index[word])
                else:
                    wids.append(word2index['UNK'])
            if make_categorical:
                data[i] = np_utils.to_categorical(wids, num_classes = num_classes)
            else:
                data[i] = wids

    pdata = pad_sequences(data, maxlen)
    return pdata

X = build_tensor('./treebank_sents.txt', num_recs, s_word2index, MAX_SEQLEN)
Y = build_tensor('./treebank_poss.txt', num_recs, t_word2index, MAX_SEQLEN, True, t_vocabsize)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.2, random_state = 42)

EMBED_SIZE  = 128
HIDDEN_SIZE = 64
BATCH_SIZE  = 32
EPOCH_SIZE  = 10

model = Sequential()
model.add(Embedding(s_vocabsize, EMBED_SIZE, input_length = MAX_SEQLEN))
model.add(SpatialDropout1D(0.2))
model.add(GRU(units=HIDDEN_SIZE, dropout = 0.2, recurrent_dropout = 0.2, return_sequences = True))
model.add(TimeDistributed(Dense(t_vocabsize)))
model.add(Activation('softmax'))
model.summary()

import keras.backend as K
def accuracy_without_padding(y_true, y_pred):
    y_true_classID = K.argmax(y_true, axis=-1)
    y_true_clipped = K.clip(y_true_classID, 0, 1)
    y_pred_classID = K.argmax(y_pred, axis=-1)
    return K.mean(K.equal(y_true_classID, y_pred_classID*y_true_clipped))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = [accuracy_without_padding])

model.fit(Xtrain, Ytrain, batch_size = BATCH_SIZE, epochs= EPOCH_SIZE, validation_data = [Xtest, Ytest])  # validation accuracy 0.99
