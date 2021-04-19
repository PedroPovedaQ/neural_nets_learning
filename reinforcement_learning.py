# -*- coding: utf-8 -*-
"""
Module Reinforcement Learning
Tutorial Link: https://alibaba-cloud.medium.com/implementing-reinforcement-learning-with-keras-e592ae123de1
"""

__author__ = "Pedro Poveda"
__version__ = "0.1.0"
__license__ = "MIT"

from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist


def main():
    """ Main entry point of the app """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print('hi', x_test)
    InputModel = Input(shape=(784,))
    EncodedLayer = Dense(32, activation='relu')(InputModel)
    DecodedLayer = Dense(784, activation='sigmoid')(EncodedLayer)
    AutoencoderModel = Model(InputModel, DecodedLayer)
    AutoencoderModel.compile(optimizer='adadelta', loss='binary_crossentropy')
    history = AutoencoderModel.fit(x_train, x_train,
                                  batch_size=256,
                                  epochs=100,
                                  shuffle=True,
                                  validation_data=(x_test, x_test))
    DecodedDigits = AutoencoderModel.predict(x_test)

    # Print Test Results
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Autoencoder Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    n=10

    # Verify Results
    plt.figure(figsize=(20, 4))
    for i in range(n):
      ax = plt.subplot(2, n, i + 1)
      plt.imshow(x_test[i].reshape(28, 28))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(DecodedDigits[i].reshape(28, 28))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
    plt.show()



if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()