"""
Project 3 - Convolutional Neural Networks for Image Classification
"""

import tensorflow as tf
from tensorflow.python.ops.gen_nn_ops import Conv2D
import hyperparameters as hp
from tensorflow.keras import layers, Sequential


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, input, training=None):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(input)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        return output


class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel, self).__init__()
        self.layer_dims = [2, 2, 2, 2]
        self.num_classes = 15

        # Optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hp.learning_rate,
            momentum=hp.momentum)

        self.stem = Sequential([
            layers.Conv2D(64, (3, 3), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        ])
        # resblock
        self.layer1 = self.build_resblock(64, self.layer_dims[0])
        self.layer2 = self.build_resblock(128, self.layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, self.layer_dims[2], stride=2)

        self.head = [
            layers.Conv2D(128, 1, 1, padding='same', activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(rate=0.4),
            layers.Dense(15, activation="softmax"),
        ]

        self.architecture = [
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3,
        ] + self.head


        # ====================================================================

    def call(self, img):
        """ Passes input image through the network. """

        for layer in self.layer_nn:
            img = layer(img)

        return img

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        for pre in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        return tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions, from_logits=False)

    def call(self, img):
        """ Passes input image through the network. """

        for layer in self.architecture:
            img = layer(img)

        return img

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        for pre in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        return tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions, from_logits=False)
