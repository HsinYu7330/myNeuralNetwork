# Recurrent Neural Network (RNN)

from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential, Model 
from keras.layers import Input, Dropout, Embedding, Dense, SimpleRNN, GlobalAveragePooling1D

# setting GPU configuration
import os
import tensorflow as tf 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(sess)

# Load Data
max_features = 20000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# pad sequences
x_train = sequence.pad_sequences(x_train, maxlen=80)
x_test = sequence.pad_sequences(x_test, maxlen=80)
print('x_train.shape', x_train.shape)
print('x_test.shape', x_test.shape)

# Build RNN model
import numpy as np

np.random.seed(5280)
inputs = Input(shape=(None,), dtype='int32')
embeddings = Embedding(max_features, 128)(inputs)
latent_vec = SimpleRNN(64, return_sequences=True)(embeddings)
latent_vec = Dropout(0.2)(latent_vec)
latent_vec = SimpleRNN(32, return_sequences=False)(latent_vec)
latent_vec = Dropout(0.2)(latent_vec)
outputs = Dense(1, activation='sigmoid')(latent_vec)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))

# Performance
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

pred = model.predict(x_test)[:,0]
pred = [1 if x >= 0.5 else 0 for x in pred] # recode probability to binary outcome
print('Precision: ', '{:.3f}'.format(precision_score(y_test, pred)))
print('Recall: ', '{:.3f}'.format(recall_score(y_test, pred)))
print('F1-Score: ', '{:.3f}'.format(f1_score(y_test, pred)))
print('Accuracy: ', '{:.3f}'.format(accuracy_score(y_test, pred)))

'''
Precision: 0.799
Recall: 0.747
F1-Score: 0.772
Accuracy: 0.779
'''