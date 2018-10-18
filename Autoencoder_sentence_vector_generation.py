# This program implements an autoencoder to generate sentence vectors

from bs4 import BeautifulSoup
import nltk
import collections
import re
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Bidirectional, LSTM, RepeatVector

docid = []
sentences = []
with open("../../Downloads/reuters21578/reut2-000.sgm","r") as fin:
    soup = BeautifulSoup(fin, 'lxml')
    reuters = soup.find_all('reuters')
    for reuter in reuters:
        docid.append(reuter.get('newid'))
    texts = soup.find_all('text')
    for text in texts:
        par = text.contents[-1].replace("\n"," ")
        last_Reuter_start = par.lower().rfind('reuter')
        if '.  ' in par[0:last_Reuter_start]:
            sents = par[0:last_Reuter_start].split(".  ")
            for sent in sents:
                if len(sent.strip())>=2 and sent.strip()[0] is not '-' and '--' not in sent.strip() \
                    and '......' not in sent.strip() and 'Note' not in sent.strip() and 'unquoted' not in sent.strip():
                    if '. ' in sent and 'U.S.' not in sent and 'U.K.' not in sent:
                        sent_parts = sent.strip().split(". ")
                        for sent_part in sent_parts:
                            if len(sent_part.strip())>2:
                                sentences.append(sent_part)
                    if '. ' not in sent and len(sent.strip())>2:
                        sentences.append(sent.strip())

def is_number(n):
    temp = re.sub("[.,-/]", "", n)
    return temp.isdigit()

word_freqs = collections.Counter()
sent_lens = []
parsed_sentence = []
for sent in sentences:
    words = nltk.word_tokenize(sent)
    parsed_words = []
    for word in words:
        if is_number(word):
            word = '9'
        word_freqs[word.lower()] += 1
        parsed_words.append(word.lower())
    sent_lens.append(len(words))
    parsed_sentence.append(" ".join(parsed_words))

VOCAB_SIZE = 5000
SEQLEN = len(word_freqs)

word2index={}
word2index['PAD'] = 0
word2index['UNK'] = 1
for v, (word, _) in enumerate(word_freqs.most_common(VOCAB_SIZE - 2)):
    word2index[word] = v + 2
index2word = {v:k for k, v in word2index.items()}

def lookup_word2id(word):
    try:
        return word2index[word]
    except KeyError:
        return word2index["UNK"]

sents_wids = np.empty((len(parsed_sentence),), dtype = list)
for i, sent in enumerate(parsed_sentence):
    sents_wids[i] = [lookup_word2id(v.lower()) for v in sent.strip().split()]
sents_padded = pad_sequences(sents_wids, max(sent_lens))
X_train, X_test = train_test_split(sents_padded, test_size = 0.3)

def load_glove_vectors(filename, word2index, EMBED_SIZE):
    embedding = np.ones((len(word2index), EMBED_SIZE))
    embedding = embedding*np.random.uniform(-1, 1, EMBED_SIZE)
    with open(filename,"r") as fin:
        for line in fin:
            cols = line.strip().split()
            word = cols[0].lower()
            if word in word2index:
                embedding[word2index[word]] = np.array([float(v) for v in cols[1:]])
    embedding[0, :] = 0
    return embedding

EMBED_SIZE = 50
DATA_DIR = "../../Downloads/glove.6B/"
embeddings = load_glove_vectors(os.path.join(DATA_DIR, "glove.6B.{:d}d.txt".format(EMBED_SIZE)), word2index, EMBED_SIZE)

BATCH_SIZE = 32
def sentence_generator(X, embeddings, BATCH_SIZE):
    while True:
        num_recs = X.shape[0]
        indics = np.random.permutation(np.arange(num_recs))
        num_batches = num_recs // BATCH_SIZE
        for i in range(num_batches):
            Xbatch = embeddings[X[indics[i*BATCH_SIZE:BATCH_SIZE+i*BATCH_SIZE], :]]
            yield Xbatch, Xbatch

train_gen = sentence_generator(X_train, embeddings, BATCH_SIZE)
test_gen = sentence_generator(X_test, embeddings, BATCH_SIZE)

HIDDEN_SIZE = 64*2
inputs = Input(shape= (max(sent_lens),EMBED_SIZE))
encoder = Bidirectional(LSTM(HIDDEN_SIZE), merge_mode = 'sum', name="encoder_lstm")(inputs)
decoder = RepeatVector(max(sent_lens))(encoder)
decoder = Bidirectional(LSTM(EMBED_SIZE, return_sequences = True), merge_mode = 'sum')(decoder)
autoencoder = Model(inputs, decoder)
autoencoder.compile(optimizer='sgd', loss='mse')

num_train_step = len(X_train)// BATCH_SIZE
num_test_step = len(X_test)// BATCH_SIZE

hist = autoencoder.fit_generator(train_gen, steps_per_epoch = num_train_step, epochs = 20, validation_data = test_gen, \
                          validation_steps=num_test_step)

encoder = Model(autoencoder.input, autoencoder.get_layer("encoder_lstm").output)
def compare_cosine_similarity(x, y):
    return np.dot(x,y) / (np.linalg.norm(x, 2) * np.linalg.norm(y, 2))

for i in range(10):
    Xtest, Ytest = next(test_gen)
    Ytest_       = autoencoder.predict(Xtest)
    Xvec         = encoder.predict(Xtest)
    Yvec         = encoder.predict(Ytest_)
    for j in range(Xvec.shape[0]):
        print(compare_cosine_similarity(Xvec[j], Yvec[j]))


