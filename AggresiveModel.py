import argparse
import pandas as pd
import numpy as np
from collections import Counter
import itertools
import os
import torch
import torch.backends.cudnn as cudnn
import torch.backends.cudnn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import unicodedata
from time import time
import string
import random
import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize, word_tokenize
from nltk import pos_tag
from pathlib import Path
import json
import gc
import re


def seed_everything(seed=786):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()


############################ Tokenizer #################################################################################
class Tokenizer:
    def __init__(self, max_features=20000, max_len=10, tokenizer=str.split):
        self.max_features = max_features
        self.tokenizer = tokenizer
        self.doc_freq = None
        self.vocab = None
        self.vocab_idx = None
        self.max_len = max_len

    def fit_transform(self, texts):
        tokenized = []
        n = len(texts)

        tokenized = [self.tokenizer(text) for text in texts]
        self.doc_freq = Counter(itertools.chain.from_iterable(tokenized))

        vocab = [t[0] for t in self.doc_freq.most_common(self.max_features)]
        vocab_idx = {w: (i + 1) for (i, w) in enumerate(vocab)}
        # doc_freq = [doc_freq[t] for t in vocab]

        # self.doc_freq = doc_freq
        self.vocab = vocab
        self.vocab_idx = vocab_idx

        result_list = []
        # tokenized = [self.tokenizer(text) for text in texts]
        for text in tokenized:
            text = self.text_to_idx(text, self.max_len)
            result_list.append(text)

        result = np.zeros(shape=(n, self.max_len), dtype=np.int32)
        for i in range(n):
            text = result_list[i]
            if len(text) > self.max_len:
                # first23 = int(0.66*self.max_len)
                # last13 = self.max_len - first23
                # result[i, :self.max_len] = text[:first23] + text[-last13:]
                result[i, :self.max_len] = text[:self.max_len]
            else:
                result[i, :len(text)] = text

        return result

    def text_to_idx(self, tokenized, max_len):
        return [self.vocab_idx[t] for i, t in enumerate(tokenized) if (t in self.vocab_idx)]

    def transform(self, texts):
        n = len(texts)
        result = np.zeros(shape=(n, self.max_len), dtype=np.int32)

        for i in range(n):
            text = self.tokenizer(texts[i])
            text = self.text_to_idx(text, self.max_len)
            if len(text) > self.max_len:
                # first23 = int(0.66*self.max_len)
                # last13 = self.max_len - first23
                # result[i, :self.max_len] = text[:first23] + text[-last13:]
                result[i, :self.max_len] = text[:self.max_len]

            else:
                result[i, :len(text)] = text

        return result

    def vocabulary_size(self):
        return len(self.vocab) + 1


############### Spell correction related stuff ##################################################################
def correction(word, vocab, key):
    "Most probable spelling correction for word."
    return max(candidates(word, vocab), key=key)


def candidates(word, vocab):
    "Generate possible spelling corrections for word."
    return (known([word], vocab) or known(edits1(word), vocab) or [word])


def known(words, vocab):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in vocab)


def edits1(word):
    "All edits that are one edit away from `word`."
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def get_coefs(line): return line[0], np.asarray(line[1:], dtype='float32')


def initialize_embeddings_glove(embedding_file, tokenizer, embed_size=300, correct_spell=False,
                                break_words=False, encoding=None):
    oov_list = []
    corrected_list = []
    word_index = tokenizer.vocab_idx
    word_freq = tokenizer.doc_freq
    nb_words = len(word_index) + 1
    embedding_matrix = np.zeros((nb_words, embed_size))

    embeddings_index = {}  # dict(get_coefs(o.split(' ')) for o in open(embedding_file, encoding="utf-8"))
    vocab = set(tokenizer.vocab)
    if encoding is not None:
        with open(embedding_file, encoding='utf-8', errors='ignore') as f:
            for line in f:
                word, arr = line.split(' ', maxsplit=1)
                if word in vocab:
                    embeddings_index[word] = np.asarray(arr.split(' '), np.float32)
    else:
        with open(embedding_file, errors='ignore') as f:
            for line in f:
                word, arr = line.split(' ', maxsplit=1)
                if word in vocab:
                    embeddings_index[word] = np.asarray(arr.split(' '), np.float32)

    valid_breaks = []
    for word, i in word_index.items():
        if word in embeddings_index:
            embedding_matrix[i] = embeddings_index[word]
        elif word.lower() in embeddings_index:
            embedding_matrix[i] = embeddings_index[word.lower()]
        else:
            if correct_spell:
                corrected = correction(word, embeddings_index, lambda x: word_freq[x] / len(x))
                if corrected in embeddings_index:
                    embedding_matrix[i] = embeddings_index[corrected]
                    corrected_list.append((word, corrected))
                    continue
                else:
                    oov_list.append(word)
                    continue
            if break_words:
                if len(word) <= 2:
                    oov_list.append(word)
                else:
                    found = False
                    if word[:-1] in embeddings_index:
                        embedding_matrix[i] = embeddings_index[word[:-1]]
                        found = True
                        continue
                    for i in range(len(word) - 2):
                        a, b = word[:i + 1], word[i + 1:]
                        if (a in embeddings_index) and (b in embeddings_index):
                            embedding_matrix[i] = 0.5 * (embeddings_index[a] + embeddings_index[b])
                            valid_breaks.append((word, a, b))
                            found = True
                            if i > len(word) // 2:
                                break
                        elif (a in embeddings_index) and (i > len(word) * 0.6):
                            embedding_matrix[i] = embeddings_index[a]
                            found = True
                            # break
                    if not found:
                        oov_list.append(word)
            else:
                oov_list.append(word)
    return embedding_matrix, oov_list, corrected_list, valid_breaks


def unicodeToAscii(series):
    return series.apply(lambda s: unicodedata.normalize('NFKC', str(s)))


STOP_WORDS = set(stopwords.words('english'))


def normalizeString(series):
    series = unicodeToAscii(series)
    return series

########## More preprocessing stripped off public kernels #########################


def clean_numbers(series):
    series = series.str.replace('[0-9]{5,}', '#####')
    series = series.str.replace('[0-9]{4}', '####')
    series = series.str.replace('[0-9]{3}', '###')
    series = series.str.replace('[0-9]{2}', '##')
    return series


def get_text_data(tr, te, max_features, max_len, embedding_file, correct_spell, break_words, tokenizer, clean_nums,
                    normalize_string, replace_misspell):
    start_time = time()
    tr = tr.copy()
    te = te.copy()
    # preprocess and  clean data
    tr["question_text"] = tr["question_text"].fillna("_#_").astype(str)
    te["question_text"] = te["question_text"].fillna("_#_").astype(str)

    if clean_nums:
        tr["question_text"] = clean_numbers(tr["question_text"])
        te["question_text"] = clean_numbers(te["question_text"])

    if normalize_string:
        tr["question_text"] = normalizeString(tr["question_text"])
        te["question_text"] = normalizeString(te["question_text"])

    if replace_misspell:
        tr["question_text"] = tr["question_text"].apply(lambda x: replace_typical_misspell(x))
        te["question_text"] = te["question_text"].apply(lambda x: replace_typical_misspell(x))

    # Tokenize Data
    print("Tokenizing data")
    tok = Tokenizer(max_features=max_features, max_len=max_len, tokenizer=tokenizer)
    X = tok.fit_transform(
        pd.concat([tr["question_text"].astype(str).fillna("na"), te["question_text"].astype(str).fillna("na")]))
    X_train = X[:len(tr), :].astype(np.int64)
    X_test = X[len(tr):, :].astype(np.int64)

    print(X_train.shape, X_test.shape)
    
    return X_train, X_test, tok
    
    
def get_embeddings(embedding_file, tokenizer, correct_spell, break_words):
    if 'wiki' in embedding_file:
        encoding=None
    else:
        encoding='utf-8'

    # Generate embedding weights
    embedding_matrix, oov_list, corr_list, breaks = initialize_embeddings_glove(embedding_file, tokenizer,
                                                                                correct_spell=correct_spell,
                                                                                break_words=break_words,
                                                                                encoding=encoding)
    print("No. of oov list", len(oov_list))
    if correct_spell:
        print("No. of corr words", len(corr_list))
    if break_words:
        print("No. of broken words", len(breaks))
    print("---------Finished generating data -----------------")
    print("Took {:2.2f} s for generating data".format(time() - start_time))
    return embedding_matrix
    

# ====================== Meta Feature extraction ===================================================================
class ExtraFeats(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(X, pd.Series):
            num_words = X.str.count("\s+").astype(int)
            num_chars = X.str.len()
            abbreviations = X.str.count("\s[A-Z0-9]{3,}\s")
            num_sent = X.str.count('[a-z][?]') + 1
            multi_puncts = X.str.count('[!?.#,$]{2,}')
            title_words = X.str.count('[A-Z]{1}[a-z]{1,}')
            mean_words = num_words / num_chars
            word_sent_rat = num_words / num_sent

            X = pd.concat(
                    [num_words, num_chars, abbreviations, num_sent, multi_puncts, title_words, mean_words,
                     word_sent_rat], axis=1)

            X = X.replace([np.inf, -np.inf], np.nan)

            return X.fillna(0)
        else:
            raise ValueError("Need pandas series as input")


def get_extra_feats(train, test, scaler):
    start_time = time()

    etr = ExtraFeats()
    X_train_extra = etr.fit_transform(train["question_text"].astype(str)).values
    X_test_extra = etr.transform(test["question_text"].astype(str)).values
    X_train_extra = scaler.fit_transform(X_train_extra).astype(np.float32)
    X_test_extra = scaler.transform(X_test_extra).astype(np.float32)
    print("Took {:2.2f} s for generating meta data".format(time() - start_time))
    return X_train_extra, X_test_extra

############################################# Extra Feats done #############################################

############################################ Model #########################################################
import inspect
import numpy as np
np.random.seed(1001)
import tensorflow as tf
tf.set_random_seed(101)
from keras.layers import (GRU, CuDNNGRU, Bidirectional, Permute, Reshape, Dense, Lambda, RepeatVector,
                          GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, merge, Embedding,
                          SpatialDropout1D, Dropout, Input, PReLU, BatchNormalization, Layer, Activation,
                          Flatten)
from keras import initializers, regularizers, constraints
from keras.models import Model
from keras.optimizers import Adam, Nadam, RMSprop
import keras.backend as K
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import copy


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


class AttentionLayer(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))),
                        (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim


class GRUClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 max_seq_len=100,
                 embed_vocab_size=100000,
                 embed_dim=300,
                 spatial_dropout=0.2,  # Spatial Dropout to be used just after embedxing layer
                 gru_dim=150,  # Hidden dimension for GRU cell
                 cudnn=True,
                 bidirectional=True,  # Whether to use bidirectional GRU cell
                 gru_layers=1,
                 attention=False,
                 single_attention_vector=True,
                 apply_before_lstm=False,
                 trainable=False,
                 add_extra_feats=True,
                 extra_feats_dim=0,
                 pooling='max_attention',  
                 # Type of pooling layer to be applied on GRU output sequences
                 # Various options for pooling layer are:
                 # 'max' : GlobalMaxPooling Layer
                 # 'mean' : GlobalAverage PoolingLayer
                 # 'attention' : Weighted attention layer
                 # 'max_attention' : Concatenation to max pooling and attention layer
                 fc_dim=256,  # Dimension for fully connected layer
                 fc_dropout=0.2,  # Dropout ot be used before fully connected layer
                 fc_layers=1,
                 prelu=True,
                 optimizer='adam',  # Optimizer to be used
                 loss="binary_crossentropy",
                 out_dim=6,
                 batch_size=256,
                 epochs=1,
                 verbose=1,
                 callbacks=None,
                 mask_zero=False,
                 # Mask zero values in embeddings layer; Zero values mostly are result of padding and/or OOV words
                 model_id=None,  # To be used for predicting if we are using checkpoints in callbacxk for our model
                 embed_kwargs={},  # Dict of keyword arguments for word embeddings layer
                 gru_kwargs={},  # Dict of keyword arguments for gru layer
                 opt_kwargs={},  # Dict of keyword arguments for optimization algo
                 fc_kwargs={}
                 ):

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def _gru_block(self, x):
        # Learn document encoding using CuDNNGRU and return hidden sequences
        if self.cudnn:
            gru_layer = CuDNNGRU(self.gru_dim, return_sequences=True, return_state=True)
        else:
            gru_layer = GRU(self.gru_dim, return_sequences=True, return_state=True)

        # Apply bidirectional wrapper if flag is True
        if self.bidirectional:
            enc = Bidirectional(gru_layer, merge_mode="sum")(x)
            x = enc[0]
            state = enc[1]
        else:
            x, state = gru_layer(x)
        return x, state

    # Taken from here: https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention_lstm.py
    # Big Thanks to Author!
    def _attention_3d_block(self, inputs):
        # inputs.shape = (batch_size, time_steps, input_dim)
        input_dim = int(inputs.shape[2])
        a = Permute((2, 1))(inputs)
        a = Reshape((input_dim, self.max_seq_len))(
            a)  # this line is not useful. It's just to know which dimension is what.
        a = Dense(self.max_seq_len, activation='softmax')(a)
        if self.single_attention_vector:
            a = Lambda(lambda x: K.mean(x, axis=1))(a)
            a = RepeatVector(input_dim)(a)
        a_probs = Permute((2, 1))(a)
        output_attention_mul = merge([inputs, a_probs], mode='mul')
        return output_attention_mul

    def _pool_block(self, x, state):
        # Pool layers
        if self.pooling == 'mean':
            x = GlobalAveragePooling1D()(x)
            x = concatenate([x, state])

        if self.pooling == 'max':
            x = GlobalMaxPooling1D()(x)
            x = concatenate([x, state])

        if self.pooling == 'attention':
            x = AttentionLayer(self.max_seq_len)(x)
            x = concatenate([x, state])

        if self.pooling == "mean_max":
            x1 = GlobalAveragePooling1D()(x)
            x2 = GlobalMaxPooling1D()(x)
            # x3 = AttentionLayer(self.max_seq_len)(x)
            x = concatenate([x1, x2, state])

        elif self.pooling == 'max_attention':
            x1 = GlobalAveragePooling1D()(x)
            x2 = GlobalMaxPooling1D()(x)
            x3 = AttentionLayer(self.max_seq_len)(x)
            x = concatenate([x1, x2, x3, state])
        return x

    def _fc_block(self, x):
        # Fully connected layer
        x = Dropout(self.fc_dropout)(x)
        x = Dense(self.fc_dim, **self.fc_kwargs)(x)
        if self.prelu:
            x = PReLU()(x)
        return x

    def _build_model(self):
        # Set input
        inp = Input(shape=(self.max_seq_len,))

        if self.add_extra_feats:
            assert self.extra_feats_dim > 0
            inp2 = Input(shape=(self.extra_feats_dim,))
        # word embedding layer
        emb = Embedding(self.embed_vocab_size, self.embed_dim, trainable=self.trainable, **self.embed_kwargs)(inp)

        if self.apply_before_lstm:
            emb = self._attention_3d_block(emb)

        # Apply spatial dropout to avoid overfitting
        x = SpatialDropout1D(self.spatial_dropout)(emb)

        for _ in range(self.gru_layers):
            x, state = self._gru_block(x)

        if (~self.apply_before_lstm) and self.attention:
            x = self._attention_3d_block(x)

        x = self._pool_block(x, state)

        if self.add_extra_feats:
            x2 = BatchNormalization()(inp2)
            x = concatenate([x, x2])

        for _ in range(self.fc_layers):
            x = self._fc_block(x)

        # Classification layer
        out = Dense(self.out_dim, activation="sigmoid")(x)

        if self.optimizer == 'adam':
            opt = Adam(**self.opt_kwargs)

        if self.optimizer == 'nadam':
            opt = Nadam(**self.opt_kwargs)

        elif self.optimizer == 'rmsprop':
            opt = RMSprop(**self.opt_kwargs)

        if self.add_extra_feats:
            model = Model(inputs=[inp, inp2], outputs=out)
        else:
            model = Model(inputs=inp, outputs=out)
        model.compile(loss=self.loss, optimizer=opt, metrics=['accuracy'])
        return model

    def fit(self, X, y, **kwargs):
        self.model = self._build_model()

        if self.callbacks:
            self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs,
                           verbose=self.verbose,
                           shuffle=True,
                           **kwargs
                           )
        else:
            self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs,
                           verbose=self.verbose,
                           shuffle=True,
                           **kwargs)
        return self

    def predict(self, X, y=None):
        if self.model:
            #if np.any(isinstance(c, ModelCheckpoint) for c in self.callbacks):
            #    self.model.load_weights("Model_" + str(self.model_id) + ".check")
            y_hat = self.model.predict(X, batch_size=2500)
        else:
            raise ValueError("Model not fit yet")
        return y_hat

def generate_save_data(train, test):
    embed_names = {0: 'glove', 1: 'para', 2: 'wiki'}
    for j, params in enumerate(PARAMS):
        X_train, X_test, tok = get_text_data(train, test, **params)
        for i, embedding_file in enumerate([GLOVE_EMBEDDING, PARA_EMBEDDING, WIKI_EMBEDDING]):
            if ((i == 0) | ((i > 0) & (j == 0))): 
                params["embedding_file"] = embedding_file
                print("------------------------------------")
                print("Generating data for {} and {} set with {}".format(embed_names[i], j, params))
                embedding_matrix = get_embeddings(embedding_file, tok, params['correct_spell'], params['break_words'])
                np.save(str(save_path / 'X_train_{}_{}.npy'.format(embed_names[i], j)), X_train)
                np.save(str(save_path / 'X_test_{}_{}.npy'.format(embed_names[i], j)), X_test)
                np.save(str(save_path / 'em_matrix_{}_{}.npy'.format(embed_names[i], j)), embedding_matrix)
                del embedding_matrix
                gc.collect()
            else:
                pass
        del X_train, X_test, tok
        gc.collect()
            
def generate_extra_data(train, test):
    scaler = MinMaxScaler()
    X_train, X_test = get_extra_feats(train, test, scaler = scaler)
    np.save(str(save_path / 'X_train_extra.npy'), X_train)
    np.save(str(save_path / 'X_test_extra.npy'), X_test)
    
    
def run_model(paramid, embed, model, lr_list):
    modelid = 0

    proc_data_path = Path('../preprocessed_data')
    save_path = Path('../output_bigrus')
    save_path.mkdir(exist_ok=True)

    embedding = embed
    modelname = "{}_{}_{}".format(embedding, paramid, modelid)
    """
    =========================================================
    Read Data
    =========================================================
    """
    X_train = np.load("{}/X_train_{}_{}.npy".format(proc_data_path, embedding, paramid))
    X_test = np.load("{}/X_test_{}_{}.npy".format(proc_data_path, embedding, paramid))
    embedding_matrix = np.load("{}/em_matrix_{}_{}.npy".format(proc_data_path, embedding, paramid))
    X_train_extra = np.load("{}/X_train_extra.npy".format(proc_data_path))
    X_test_extra = np.load("{}/X_test_extra.npy".format(proc_data_path))

    X_train_all = [X_train, X_train_extra]
    X_test_all = [X_test, X_test_extra]
    gc.collect()

    def schedule(epoch):
        if epoch < len(lr_list):
            print(f"lr - {lr_list[epoch]}")
            return lr_list[epoch]
        print("lr - 0.0001")
        return 0.0001


    np.random.seed(1001)
    callbacks=[LearningRateScheduler(schedule)]
    
    check_filename="{}.check".format(modelname)
    start_time = time()
    model.embed_vocab_size = embedding_matrix.shape[0]
    model.extra_feats_dim = X_test_extra.shape[1]
    model.max_seq_len = PARAMS[paramid]["max_len"]
    model.embed_kwargs = {"weights": [embedding_matrix], "mask_zero": False}
    model.fit(X_train_all, y, callbacks=callbacks)
    y_test11 = model.predict(X_test_all)
    return y_test11
    
################################################## Model done #######################################

################################################## utility functions ################################


if __name__ == "__main__":
    data_path = Path('../input')
    save_path = Path('../preprocessed_data')
    save_path.mkdir(exist_ok=True)
    
    TRAIN_FILE = str(data_path / 'train.csv')
    TEST_FILE = str(data_path / 'test.csv')
    GLOVE_EMBEDDING = str(data_path / 'embeddings/glove.840B.300d/glove.840B.300d.txt')
    PARA_EMBEDDING = str(data_path / 'embeddings/paragram_300_sl999/paragram_300_sl999.txt')
    WIKI_EMBEDDING = str(data_path / 'embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec')
    
    PARAMS1 = {
    "max_features": 250000,
    "max_len": 40,
    "embedding_file": GLOVE_EMBEDDING,
    "correct_spell": True,
    "break_words": False,
    "tokenizer": wordpunct_tokenize,
    "clean_nums": False,
    "normalize_string": True,
    "replace_misspell": False
    }
    

    PARAMS6 = {
    "max_features": 250000,
    "max_len": 30,
    "embedding_file": GLOVE_EMBEDDING,
    "correct_spell": True,
    "break_words": True,
    "tokenizer": word_tokenize,
    "clean_nums": True,
    "normalize_string": True,
    "replace_misspell": False
    }
    
    PARAMS = [PARAMS1, PARAMS6]
    
    """
    =========================================================
    Read Data
    =========================================================
    """
    start_time = time()
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    y = train["target"].values.reshape(-1, 1)
    print("Took {:2.2f} s for reading data".format(time() - start_time))
    start_time = time()

    """
    =========================================================
    Generate Data
    =========================================================
    """
    generate_save_data(train, test)
    gc.collect()
    
    '''
    =========================================================
    Extra Feats
    =========================================================
    '''
    generate_extra_data(train, test)
    gc.collect()
    del train
    '''
    =========================================================
    Run Model
    =========================================================
    '''
    MODEL_PARAMS1 = {
            "max_seq_len": 30,
            "embed_vocab_size": 250000,
            "embed_dim": 300,
            "trainable": False,
            "spatial_dropout": 0.00,
            "gru_dim" : 400,
            "cudnn" : True,
            "bidirectional" : True,
            "gru_layers": 1,
            "add_extra_feats": True,
            "extra_feats_dim" : 8,
            "attention": False,
            "single_attention_vector":False,
            "apply_before_lstm":False,
            "pooling": 'mean_max',
            "fc_dim": 256,
            "fc_dropout": 0.2,
            "fc_layers": 2,
            "optimizer": 'adam',
            "out_dim": 1,
            "batch_size": 1024,
            "epochs": 4,
            "callbacks": [],
            "loss": "binary_crossentropy",
            "model_id": "model_1",
            "mask_zero":False,
            "embed_kwargs": None,
            "verbose": 2,
            "opt_kwargs": {"lr": 0.001, "decay": 1e-10, "clipvalue": 5},
            }
    model1 = GRUClassifier(**MODEL_PARAMS1)
    
    lr_list = [0.001, 0.001, 0.001, 0.0002]
    
    y_test11 = run_model(0, 'glove', model1, lr_list)
    y_test12 = run_model(1, 'glove', model1, lr_list)
    y_test21 = run_model(0, 'wiki', model1, lr_list)
    y_test31 = run_model(0, 'para', model1, lr_list)
    del model1
    MODEL_PARAMS2 = {
            "max_seq_len": 30,
            "embed_vocab_size": 250000,
            "embed_dim": 300,
            "trainable": False,
            "spatial_dropout": 0.0,
            "gru_dim" : 600,
            "cudnn" : True,
            "bidirectional" : True,
            "gru_layers": 1,
            "add_extra_feats": True,
            "extra_feats_dim" : 8,
            "attention": False,
            "single_attention_vector":False,
            "apply_before_lstm":False,
            "pooling": 'mean_max',
            "fc_dim": 256,
            "fc_dropout": 0.0,
            "fc_layers": 1,
            "optimizer": 'adam',
            "out_dim": 1,
            "batch_size": 1024,
            "epochs": 1,
            "callbacks": [],
            "loss": "binary_crossentropy",
            "model_id": "model_1",
            "mask_zero": False,
            "embed_kwargs": None,
            "verbose": 2,
            "opt_kwargs": {"lr": 0.001, "decay": 0, "clipvalue": 5},
            }
    model2 = GRUClassifier(**MODEL_PARAMS2)
    y_test41 = run_model(0, 'glove', model2, lr_list)
    
    y_test = y_test11*0.3 + y_test12*0.175 + y_test21*0.175 + y_test31*0.15 + y_test41*0.20
    test_preds: pd.DataFrame = test[['qid']]
    test_preds.loc[:, "prediction"] = (y_test > 0.34).astype(int)
    test_preds.to_csv("submission.csv", index=False)
