'''
Similar to the architecture in this paper: 
https://arxiv.org/abs/1608.06993 
Adapted to a truncated, 1D version for our purposes.
Allows for a couple different 'modes'
A normal version, one with a longer stem, or
one with separable depthwise convolutions 
'''

# imports
from keras.layers import Conv2D, BatchNormalization, Dense, Dropout, Activation, AveragePooling1D, GaussianNoise
from keras.layers import Input, Concatenate, Flatten, Conv1D, MaxPooling1D, Reshape, SeparableConv1D
from keras.layers import GlobalMaxPooling1D, Lambda, GlobalAveragePooling1D
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras import backend as K
import numpy as np
import os


# 1D Densenet
class OneDimDensenet():
    def __init__(self, start_target_size = (672, 4), mode='vanilla'):
        
        #set of conv blocks wrapper
        def conv_block(x, dim):
            x1 = Conv1D(dim, kernel_size=1, strides=1, padding='same', activation='relu')(x)
            x1 = BatchNormalization()(x1)
            x1 = Conv1D(dim, kernel_size=3, strides=1, padding='same', activation='relu')(x1)
            x1 = BatchNormalization()(x1)
            return x1

        # set of separable conv wrapper
        def sepa_conv_block(x, dim):
            x1 = SeparableConv1D(dim, kernel_size=1, strides=1, padding='same', activation='relu')(x)
            x1 = BatchNormalization()(x1)
            x1 = SeparableConv1D(dim, kernel_size=3, strides=1, padding='same', activation='relu')(x1)
            x1 = BatchNormalization()(x1)
            return x1

        # dense block wrapper
        def dense_block(inlayer, convs, dims):
            conv_list = []
            ministem = conv_block(inlayer, dims) if mode != 'separable' else sepa_conv_block(inlayer, dims)
            ministem = BatchNormalization()(ministem)
            conv_list.append(ministem)
            ministem = conv_block(conv_list[0], dims) if mode != 'separable' else sepa_conv_block(conv_list[0], dims)
            ministem = BatchNormalization()(ministem)
            conv_list.append(ministem)
            for _ in range(convs-2):
                x = Concatenate()([layer for layer in conv_list])
                x = conv_block(x, dims) if mode != 'separable' else sepa_conv_block(x, dims)
                x = BatchNormalization()(x)
                conv_list.append(x)
            return conv_list[-1]


        ## build our model
        # stem
        inputs = Input(shape=start_target_size)
        x = GaussianNoise(0.3)(inputs)
        x = Conv1D(512, kernel_size=7, strides=2, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=3, strides=2)(x)
            
        # dense block 1
        d1 = dense_block(x, 6, 64)

        #transition
        t = Conv1D(64, kernel_size=1, strides=1, padding='same', activation='relu')(d1)
        t = MaxPooling1D(pool_size=2, strides=2)(t)

        # dense block 2
        d2 = dense_block(t, 12, 64)

        # optional depth, doesn't seem to help
        '''
        #transition
        t2 = Conv1D(64, kernel_size=1, strides=1, padding='same', activation='relu')(d2)
        t2 = AveragePooling1D(pool_size=2, strides=2)(t2)

        # dense block 2
        d3 = dense_block(t2, 6, 64)
        '''

        # exit stem
        fc = Conv1D(64, kernel_size=1, strides=1, padding='same', activation='relu')(d2)
        if mode == 'long stem':
            fc = MaxPooling1D(pool_size=2, strides=2)(fc)
        fc = Flatten()(fc)
        fc = Dense(1024, activation='relu')(fc)
        fc = Dropout(0.5)(fc)
        predictions = Dense(1, activation='sigmoid')(fc)

        model = Model(inputs=inputs, outputs=predictions)
        model.compile(loss='binary_crossentropy',
                      optimizer= SGD(lr=1e-3, momentum=0.9),
                      metrics=['binary_accuracy'])
        model.summary()

        self.model = model
        self.mode = mode
        

    def load_data(self, 
                  batch_size = 16, 
                  x_path = 'D:/Projects/iSynPro/iSynPro/DanQCNNLSTM/x_train.npy', 
                  y_path = 'D:/Projects/iSynPro/iSynPro/DanQCNNLSTM/y_train.npy'):
        self.batch_size = batch_size
        self.x_train = np.load(x_path)
        self.y_train = np.load(y_path)

    def train(self):
        # save path, callbacks
        local_path = os.path.dirname(os.path.abspath(__file__))
        if self.mode == 'vanilla':
            subdir_name = '1ddense_weights'
        elif self.mode == 'separable':
            subdir_name = '1ddense_separable_weights'
        else:
            subdir_name = '1ddense_longstem_weights'
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
    model_1d = OneDimDensenet(mode = 'vanilla')
    model_1d.load_data()
    model_1d.train()

if __name__ == '__main__':
    main()
