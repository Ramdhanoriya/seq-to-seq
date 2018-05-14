import  pickle
from keras.models import model_from_json
import numpy as np
from nltk import word_tokenize
from numpy import array
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model
from keras.preprocessing import sequence
from keras.layers import Dense


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



# one hot encode
def one_hot_encode(X, max_int):
    Xenc = list()
    for seq in X:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        Xenc.append(pattern)

    return Xenc


# invert encoding
def invert(seq):
    strings = list()
    for pattern in seq:
        string = int_to_word_input[np.argmax(pattern)]
        if(string!="padd"):
            strings.append(string)
        else:
            return ' '.join(strings)





# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    # encode
    state = infenc.predict(source)
    # start of sequence input
    target_seq = array(one_hot_encode(array([[word_to_int_input["_"]]]),encoded_length))

    # collect predictions
    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state)


        # store prediction
        output.append(yhat[0,0,:])

        # update state
        state = [h, c]
        # update target sequence
        target_seq = yhat
    return array(output)



classifier_f = open("word_to_int_input.pickle", "rb")
word_to_int_input = pickle.load(classifier_f)
classifier_f.close()


classifier_f = open("int_to_word_input.pickle", "rb")
int_to_word_input = pickle.load(classifier_f)
classifier_f.close()

encoded_length=len(word_to_int_input)

train, infenc, infdec = define_models(encoded_length, encoded_length, 128)

infenc.load_weights("model_enc.h5")
infdec.load_weights("model_dec.h5")


while True:

    input_data=input().lower()

    input_data=word_tokenize(input_data)

    input_data=[word_to_int_input[word] for word in input_data if word.isalpha()]

    input_data=np.array([input_data])

    input_data = sequence.pad_sequences(input_data, maxlen=10,padding='post')

    input_data=one_hot_encode(input_data,encoded_length)

    input_data=array(input_data)

    target = predict_sequence(infenc, infdec, input_data, 10, encoded_length)

    print(invert(target))

