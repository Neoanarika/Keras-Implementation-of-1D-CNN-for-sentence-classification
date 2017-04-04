# The model basically classifies between positive and negative reviews 
from __future__ import print_function
import numpy as np
import gc;
np.random.seed(1337)  # for reproducibility
from keras.models import Model
from keras.preprocessing import sequence
from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from sklearn.metrics import log_loss
# set parameters:
max_features = 5000  # vocabulary size
maxlen = 100  # maximum length of the review
batch_size = 32
embedding_dims = 20
ngram_filters = [3, 5, 7]
nb_filter = 1200  # number of filters for each ngram_filter
nb_epoch = 3

# prepare data
print('Loading data...')
#Split the data
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print(y_test)
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# define model
main_input = Input(shape=(maxlen,), dtype='float32', name='main_input')
#Convert a sentence into a vector
embedding = Embedding(max_features, embedding_dims, input_length=maxlen, name='embedding')(main_input)
x = Dropout(0.,name='dropout')(embedding)
n_grams = []
#Basically the CNN
for n_gram in ngram_filters:
  n_gram_output = Convolution1D(nb_filter=nb_filter,
               filter_length=n_gram,
               border_mode='valid',
               activation='relu',
               subsample_length=1,
               input_dim=embedding_dims,
               input_length=maxlen)(x)
  n_gram_output = MaxPooling1D(pool_length=maxlen - n_gram + 1)(n_gram_output)
  n_gram_output = Flatten()(n_gram_output)
  n_grams.append(n_gram_output)

y = Dropout(0.)(n_grams)
y = concatenate(y)
output = Dense(1, activation='sigmoid', input_dim=nb_filter * len(ngram_filters),name='output')(y)
print(output)
model = Model(main_input,output)
print(model.summary())

#Load or save model
try:
  json_file = open('model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  # load weights into new model
  model.load_weights("model.h5")
  print("Loaded model from disk")
except:
  #Train model
  model.compile(loss={'output': 'binary_crossentropy'}, optimizer='rmsprop',metrics=['accuracy'])
  model.fit({'main_input': X_train},{'output': y_train},
            batch_size=batch_size,
            nb_epoch=nb_epoch)
  # serialize model to JSON
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
      json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("model.h5")
  print("Saved model to disk")

#Evaluation of the model
acc = log_loss(y_test,np.array(model.predict({'main_input': X_test}, batch_size=int(batch_size))))
print('Test accuracy:', acc)