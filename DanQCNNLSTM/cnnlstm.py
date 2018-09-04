'''
Based on the DanQ neural network for DNA classification (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4914104/),
these are a series of experiments designed to test what is the optimum architecture 
for our given problem of DNA promoter classification given limited data,
while still remaining relatively true to the general concept of the DanQ CNN-LSTM.
Parameters we examine are stride of pooling, convolution kernel size, 
type of convolution used, and number of convolutional layers. 
'''

# imports
from keras.layers import Conv2D, BatchNormalization, AveragePooling2D, Dense, Dropout, Activation, AveragePooling1D, Bidirectional, MaxPooling2D, GaussianNoise
from keras.layers import Input, Concatenate, Flatten, Embedding, CuDNNLSTM, Conv1D, MaxPooling1D, LSTM, StackedRNNCells, LSTMCell, Reshape, TimeDistributed, SeparableConv1D
from keras.layers import RepeatVector, Permute, merge, multiply, GlobalMaxPooling1D, Lambda, BatchNormalization, GlobalAveragePooling1D
from keras.layers.merge import Multiply
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras import backend as K
import numpy as np
import os


# 1D Densenet
class CNNLSTM():
    def __init__(self, start_target_size = (672, 4), mode = 'one-hot', n_convs = 1, pool = 8, stride = 8):
        
        inputs = Input(shape=start_target_size)
        x = GaussianNoise(0.3)(inputs) if mode == 'one-hot' else Embedding(input_dim = 5, output_dim=16, mask_zero=False, input_length=start_target_size)(inputs)
        x = Conv1D(128, kernel_size=16, strides=1, padding='same', activation='relu')(x)
        if n_convs == 2:
            x2 = Conv1D(128, kernel_size=16, strides=1, padding='same', activation='relu')(x)
            x = Concatenate()([x, x2])
        if n_convs == 3:
            x2 = Conv1D(128, kernel_size=16, strides=1, padding='same', activation='relu')(x)
            x3 = Concatenate()([x, x2])
            x3 = Conv1D(128, kernel_size=16, strides=1, padding='same', activation='relu')(x3)
            x = Concatenate()([x, x2, x3])
        x = MaxPooling1D(pool_size=pool, strides=stride)(x)
        x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=predictions)
        model.compile(loss='binary_crossentropy',
                      optimizer= SGD(lr=1e-3, momentum=0.9),
                      metrics=['binary_accuracy'])
        model.summary()

        self.model = model  
        self.target_size = start_target_size  
        self.mode = mode
        self.convs = n_convs
        self.pool_size = pool
        self.pool_stride = stride    

    # load data that is shape (samples, sequence_length, 4)
    def load_data(self, 
                  batch_size = 16, 
                  x_path = 'D:/Projects/iSynPro/iSynPro/DanQCNNLSTM/x_train.npy', 
                  y_path = 'D:/Projects/iSynPro/iSynPro/DanQCNNLSTM/y_train.npy'):
        self.batch_size = batch_size
        self.x_train = np.load(x_path)
        self.y_train = np.load(y_path)
        print('Data loaded successfully...')
        if self.x_train.shape[1] != self.target_size[0] or self.x_train.shape[2] != self.target_size[1]:
            print('Warning: Data and Model dimensions are mismatched. Please modify model start_target_size.')
        

    def train(self):
        # save path, callbacks
        local_path = os.path.dirname(os.path.abspath(__file__))
        subdir_name = 'cnnlstm_{}_{}convs_{}pool_{}poolstride'.format(self.mode, self.convs, self.pool_size, self.pool_stride)
        save_path = os.path.join(local_path, subdir_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        lr_descent = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.5,
                                       patience=5,
                                       verbose=1,
                                       mode='auto',
                                       epsilon=0.0001,
                                       cooldown=1,
                                       min_lr=1e-6)

        save_model = ModelCheckpoint(os.path.join(save_path, 'weights-{epoch:02d}-{val_loss:.2f}.hdf5'),
                                     monitor='val_loss',
                                     verbose=1, 
                                     save_best_only=True,
                                     save_weights_only=False,
                                     mode='auto',
                                     period=1)

        csv_logger = CSVLogger(os.path.join(save_path, 'training_history.csv'), separator=',', append=False)


        # train model
        self.model.fit(self.x_train,
                    self.y_train,
                    batch_size=self.batch_size, 
                    epochs=30,
                    shuffle=True,
                    verbose=2, 
                    validation_split=0.1,
                    callbacks = [save_model, csv_logger])

def main():
    model = CNNLSTM(mode = 'one-hot', n_convs = 2, pool_size = 8, pool_stride = 8)
    model.load_data()
    model.train()

if __name__ == '__main__':
    main()
