# Self-Attention

from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential, Model 
from keras.layers import Input, Dropout, Embedding, Dense, GlobalAveragePooling1D
from keras.engine.topology import Layer
from keras import backend as K

# setting GPU configuration
import os
import tensorflow as tf 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(sess)

'''
Self-Attention Modules
'''
class Self_Attention(Layer):
    
    def __init__(self, nb_head, size_per_head, **kwargs):
        
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Self_Attention, self).__init__(**kwargs)
        
    
    def build(self, input_shape):
        
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel', shape=(3, input_shape[2], self.output_dim), initializer='uniform', trainable=True)
        super(Self_Attention, self).build(input_shape)
        
    def call(self, x):
        
        Q_seq = K.dot(x, self.kernel[0])
        K_seq = K.dot(x, self.kernel[1])
        V_seq = K.dot(x, self.kernel[2])
        
        # rehsape [B, N, HS] --> [B, N, H, S]
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        
        # transpose [B, N, H, S] --> [B, H, N, S]
        Q_seq = K.permute_dimensions(Q_seq, [0,2,1,3])
        K_seq = K.permute_dimensions(K_seq, [0,2,1,3])
        V_seq = K.permute_dimensions(V_seq, [0,2,1,3])
        
        # Attention
        QK = tf.keras.backend.batch_dot(Q_seq, K_seq, axes=[3,3]) # [B, H, N, N]
        QK = QK / (self.size_per_head**0.5)
        QK = K.softmax(QK)

        z = tf.keras.backend.batch_dot(QK, V_seq, axes=[3,2]) # [B, H, N, S_v]
        z = K.permute_dimensions(z, [0,2,1,3]) # [B, H, N, S_v] --> [B, N, H, S_v]
        z = K.reshape(z, (-1, K.shape(z)[1], self.output_dim))
        
        return z
    
    def compute_output_shape(self, input_shape):
        
        return (input_shape[0], input_shape[1], self.output_dim)

# Load Data
max_features = 20000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# pad sequences
x_train = sequence.pad_sequences(x_train, maxlen=80)
x_test = sequence.pad_sequences(x_test, maxlen=80)
print('x_train.shape', x_train.shape)
print('x_test.shape', x_test.shape)


# Build Self-Attention model
import numpy as np

np.random.seed(5280)
inputs = Input(shape=(None,), dtype='int32')
embeddings = Embedding(max_features, 128)(inputs)
latent_vec = Self_Attention(4,16)(embeddings) # 4 heads, 16 size per head
latent_vec = Dropout(0.2)(latent_vec)
latent_vec = Self_Attention(4,8)(embeddings)
latent_vec = Dropout(0.2)(latent_vec)
latent_vec = GlobalAveragePooling1D()(latent_vec)
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
Precision: 0.801
Recall: 0.780
F1-Score: 0.790
Accuracy: 0.793
'''