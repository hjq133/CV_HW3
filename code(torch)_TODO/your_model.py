"""
Project 3 - Convolutional Neural Networks for Image Classification
"""

import tensorflow as tf
import hyperparameters as hp
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense
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

        # resnet-18
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
            # layers.Dropout(0.2)  # special drop out layer
        ])
        # resblock
        self.layer1 = self.build_resblock(64, self.layer_dims[0])
        self.layer2 = self.build_resblock(128, self.layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, self.layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, self.layer_dims[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(self.num_classes)

        # TODO: Build your own convolutional neural network, using Dropout at
        #       least once. The input image will be passed through each Keras
        #       layer in self.architecture sequentially. Refer to the imports
        #       to see what Keras layers you can use to build your network.
        #       Feel free to import other layers, but the layers already
        #       imported are enough for this assignment.
        #
        #       Remember: Your network must have under 15 million parameters!
        #       You will see a model summary when you run the program that
        #       displays the total number of parameters of your network.
        #
        #       Remember: Because this is a 15-scene classification task,
        #       the output dimension of the network must be 15. That is,
        #       passing a tensor of shape [batch_size, img_size, img_size, 1]
        #       into the network will produce an output of shape
        #       [batch_size, 15].
        #
        #       Note: Keras layers such as Conv2D and Dense give you the
        #             option of defining an activation function for the layer.
        #             For example, if you wanted ReLU activation on a Conv2D
        #             layer, you'd simply pass the string 'relu' to the
        #             activation parameter when instantiating the layer.
        #             While the choice of what activation functions you use
        #             is up to you, the final layer must use the softmax
        #             activation function so that the output of your network
        #             is a probability distribution.
        #
        #       Note: Flatten is a very useful layer. You shouldn't have to
        #             explicitly reshape any tensors anywhere in your network.
        #
        # ====================================================================

        self.architecture = [
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool,
            self.fc,
            layers.Activation('softmax'),
        ]

        # ====================================================================

    def call(self, img):
        """ Passes input image through the network. """

        for layer in self.architecture:
            img = layer(img)

        return img

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        # may down sample
        res_blocks.add(BasicBlock(filter_num, stride))
        # just down sample one time
        for pre in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        return tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions, from_logits=False)


# class YourModel(tf.keras.Model):
#     """ Your own neural network model. """
#
#     def __init__(self):
#         super(YourModel, self).__init__()
#
#         # Optimizer
#         self.optimizer = tf.keras.optimizers.RMSprop(
#             learning_rate=hp.learning_rate,
#             momentum=hp.momentum)
#
#         Sequential([
#             layers.Conv2D(64, (3, 3), strides=(1, 1)),
#             layers.Activation('relu'),
#             layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
#
#             layers.Conv2D(128, (3, 3), strides=(2, 2)),
#             layers.Activation('relu'),
#             layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
#
#             layers.Flatten(),
#             layers.Dense(128),
#             layers.Activation('relu'),
#             layers.Dropout(0.2),
#             layers.Dense(10, 5)
#         ])
#
#         self.architecture = [
#
#         ]
#
#         # ====================================================================
#
#     def call(self, img):
#         """ Passes input image through the network. """
#
#         for layer in self.architecture:
#             img = layer(img)
#
#         return img
#
#     @staticmethod
#     def loss_fn(labels, predictions):
#         """ Loss function for the model. """
#
#         return tf.keras.losses.sparse_categorical_crossentropy(
#             labels, predictions, from_logits=False)
