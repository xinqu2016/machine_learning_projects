# This programs builds a memory network for answering questions with test accuracy close to 1
import os
import collections
import nltk
import itertools
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers import Input, Dot, Add, Permute, Concatenate, Dense, LSTM, Dropout, GlobalAveragePooling1D, Flatten
from keras.layers import Conv2D, MaxPooling2D, Reshape, SpatialDropout2D
from keras.layers.merge import add
from keras.layers.embeddings import Embedding
from keras.backend import permute_dimensions
from keras.models import Model
import numpy as np

DATA_DIR = "../../Downloads/tasks_1-20_v1-2/en-10k/"
TRAIN_FILE = os.path.join(DATA_DIR, "qa1_single-supporting-fact_train.txt")
TEST_FILE = os.path.join(DATA_DIR, "qa1_single-supporting-fact_test.txt")

def get_data(infile):
    stories, questions, answers = [], [], []
    story_text = []
    with open(infile, "r") as fin:
        for line in fin:
            lno, text = line.encode('utf-8').strip().split(maxsplit=1)
            text = text.decode('utf-8')
            if '\t' in text:
                question, answer, _ = text.split('\t')
                stories.append(story_text)
                questions.append(question)
                answers.append(answer)
                story_text = []
            else:
                story_text.append(text)
    return stories, questions, answers

data_train = get_data(TRAIN_FILE)
data_test  = get_data(TEST_FILE)

def build_vocab(data_train, data_test):
    counter = collections.Counter()
    for stories, questions, answers in [data_train, data_test]:
        for story in stories:
            for sent in story:
                for word in nltk.word_tokenize(sent):
                    counter[word.lower()] += 1
        for question in questions:
            for word in nltk.word_tokenize(question):
                counter[word.lower()] += 1
        for answer in answers:
            for word in nltk.word_tokenize(answer):
                counter[word.lower()] += 1
    word2index = {word:j+1 for j, (word, _) in enumerate(counter.most_common())}
    word2index['PAD'] = 0
    index2word = {v:k for k, v in word2index.items()}
    return word2index, index2word

word2index, index2word = build_vocab(data_train, data_test)

vocab_size = len(word2index)

def get_maxlens(data_train, data_test):
    story_maxlen, question_maxlen = 0, 0
    for stories, questions, _ in [data_train, data_test]:
        for story in stories:
            story_len = 0
            for sent in story:
                words = nltk.word_tokenize(sent)
                story_len +=len(words)
            story_maxlen = max(story_maxlen, story_len)
        for question in questions:
            question_len = len(nltk.word_tokenize(question))
            question_maxlen = max(question_maxlen, question_len)
    return story_maxlen, question_maxlen

story_maxlen, question_maxlen = get_maxlens(data_train, data_test)

def vectorize(data, word2index, story_maxlen, question_maxlen):
    Xs, Xq, Y = [], [], []
    stories, questions, answer  = data
    for i in range(len(stories)-2):
        #xs = [[[word2index[w.lower()] for w in nltk.word_tokenize(s)] for s in story] for story in stories[i:i+3]]
        #xs = list(itertools.chain.from_iterable(itertools.chain.from_iterable(xs)))
        xs = [word2index[w.lower()] for story in stories[i:i+3] for s in story for w in nltk.word_tokenize(s)]
        xq = [word2index[w.lower()] for w in nltk.word_tokenize(questions[i+2])]
        Xs.append(xs)
        Xq.append(xq)
        Y.append(word2index[answer[i+2].lower()])
    return pad_sequences(Xs, maxlen=story_maxlen), pad_sequences(Xq, maxlen=question_maxlen), \
           np_utils.to_categorical(Y, num_classes=len(word2index))

story_maxlen = 3*story_maxlen

Xstrain, Xqtrain, Ytrain = vectorize(data_train, word2index, story_maxlen, question_maxlen)
Xstest, Xqtest, Ytest    = vectorize(data_test, word2index, story_maxlen, question_maxlen)

Inputq      = Input(shape=(question_maxlen,))
Inputs      = Input(shape=(story_maxlen,))
Embed       = Embedding(input_dim=vocab_size, output_dim=64)
Embedq      = Embed(Inputq)
Embedq      = Dropout(0.3)(Embedq)
Embeds      = Embed(Inputs)
Embeds      = Dropout(0.3)(Embeds)
sq_matching = Dot(axes=(2,2), name="dot_product")([Embeds, Embedq])

answer      = LSTM(units=128, name="LSTM")(sq_matching)
answer      = Dropout(0.5)(answer)
answer      = Dense(units=vocab_size, activation='softmax')(answer)
model       = Model(inputs=[Inputq, Inputs], outputs=answer)

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit([Xqtrain, Xstrain], [Ytrain], batch_size = 32, epochs=20, validation_data=([Xqtest, Xstest], [Ytest]))

model_1 = Model(model.input, model.get_layer("dot_product").output)
Ypred_1 = model_1.predict([Xqtest, Xstest], batch_size = 32)

model_3 = Model(model.input, model.get_layer("LSTM").output)
Ypred_3 = model_3.predict([Xqtest, Xstest], batch_size = 32)

Ypred = model.predict([Xqtest, Xstest], batch_size = 32)

pred_answer = [index2word[x] for x in np.argmax(Ypred,axis=1)]

true_answer = [index2word[x] for x in np.argmax(Ytest,axis=1)]

true_story  = [" ".join([index2word[x] for x in Xstest[y]]) for y in range(1000-2)]

true_question = [" ".join([index2word[x] for x in Xqtest[y]]) for y in range(1000-2)]

for i in range(1000-2):
    #print(true_story[i], true_question[i], true_answer[i], pred_answer[i])
    if true_answer[i] is not pred_answer[i]:
        print(i)

weights = model.get_weights()

x_temp = [sum(Ypred_3[0]*weights[5][:,i])+weights[6][i] for i in range(22)]

for i in range(22):
    print(stats.mstats.pearsonr(weights[0][i], weights[1][i])[0])
    #print(weights[5][:,i].std())
