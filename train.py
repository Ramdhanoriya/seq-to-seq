from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
import pickle
from keras.preprocessing import sequence

# one hot encode
def one_hot_encode(y, max_int):
    yenc = list()
    for seq in y:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        yenc.append(pattern)
    return yenc

# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model


dataX=[]
dataY=[]
X1=[]
X2=[]
Y=[]

# number of timesteps for encoder and decoder
n_in = 10
n_out = 20

# load dataset
data=open("dataset.txt","r").read().lower()
for i in data.split("\n\n"):
    a=i.split("\n")
    question=a[0]
    answer=a[1]
    dataX.append(question)
    dataY.append(answer)

# creating dict for integer encoding
words=open("words.txt").read().split()
word_to_int_input = dict((c, i+2) for i, c in enumerate(words))
int_to_word_input = dict((i+2, c) for i, c in enumerate(words))
word_to_int_input.update({"padd":0})
int_to_word_input.update({0:"padd"})
word_to_int_input.update({"_":1})
int_to_word_input.update({1:"_"})

encoded_length = len(word_to_int_input)
# integer encoding
for sentence in dataX:
    sentence=word_tokenize(sentence)
    sentence = [word for word in sentence if word.isalpha()]
    X1.append([word_to_int_input[word] for word in sentence])
for sentence in dataY:
    sentence=word_tokenize(sentence)
    sentence = [word for word in sentence if word.isalpha()]
    Y.append([word_to_int_input[word] for word in sentence])
for sentence in dataY:
    sentence=word_tokenize(sentence)
    sentence = [word for word in sentence if word.isalpha()]
    sentence.insert(0,"_")
    sentence=sentence[0:len(sentence)-1]
    X2.append([word_to_int_input[word] for word in sentence])

# zero padding
X1 = sequence.pad_sequences(X1, maxlen=n_in,padding='post')
X2 = sequence.pad_sequences(X2, maxlen=n_out,padding='post')
Y = sequence.pad_sequences(Y, maxlen=n_out,padding='post')

# one hot encode
X1=one_hot_encode(X1,encoded_length)
X2=one_hot_encode(X2,encoded_length)
Y=one_hot_encode(Y,encoded_length)
X1=array(X1)
X2=array(X2)
Y=array(Y)

# define model
train, infenc, infdec = define_models(encoded_length, encoded_length, 128)
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(train.summary())

# train model
train.fit([X1, X2], Y, epochs=1500)

# saving model
infenc.save_weights("model_enc.h5")
print("Saved model to disk")
infdec.save_weights("model_dec.h5")
print("Saved model to disk")

# saving integer encodings dict
save_word_features = open("word_to_int_input.pickle","wb")
pickle.dump(word_to_int_input, save_word_features)
save_word_features.close()
save_word_features = open("int_to_word_input.pickle","wb")
pickle.dump(int_to_word_input, save_word_features)
save_word_features.close()

