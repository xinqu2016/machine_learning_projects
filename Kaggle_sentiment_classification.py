# This program applies word embedding to a Kaggle sentiment classification problem.

import collections
import nltk
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Dense, SpatialDropout1D, Conv1D, GlobalMaxPooling1D

INPUT_FILE      = "../../Downloads/all/training.txt"
VOCAB_SIZE      = 5000
EMBBEDING_SIZE  = 300
FILTER_NUM      = 256
WORD_NUM        = 3
BATCH_SIZE      = 64
EPOCH_NUM       = 10

counter = collections.Counter()
xs, ys = [], []
maxlen = 0
with open(INPUT_FILE,'r') as fin:
    for line in fin:
        label, sentence = line.strip().split("\t")
        ys.append(int(label))
        words = [x.lower() for x in nltk.word_tokenize(sentence)]
        maxlen = max(maxlen,len(words))
        for word in words:
            counter[word] += 1

word2index = collections.defaultdict()
for N, pairs in enumerate(counter.most_common(VOCAB_SIZE)):
    word2index[pairs[0]] = N + 1
    vocab_size = VOCAB_SIZE + 1
index2word = {v:w for w, v in word2index.items()}

with open(INPUT_FILE,'r') as fin:
    for line in fin:
        _, sentence  = line.strip().split("\t")
        xs.append([word2index[x.lower()] for x in nltk.word_tokenize(sentence)])

X  = pad_sequences(xs, maxlen=maxlen)
Y  = np_utils.to_categorical(ys)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 43)

model = Sequential()
model.add(Embedding(input_dim = VOCAB_SIZE, output_dim = EMBBEDING_SIZE, input_length = maxlen ))
model.add(SpatialDropout1D(0.2))
model.add(Conv1D(filters = FILTER_NUM, kernel_size = WORD_NUM, activation = "relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(2,activation='softmax'))
model.summary()

model.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = EPOCH_NUM,
                    validation_data = (X_test, Y_test))  # Accuracy for the test data is 0.99

