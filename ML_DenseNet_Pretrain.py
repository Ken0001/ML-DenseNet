"""
    ML-DenseNet with imagenet weight
"""

import tensorflow as tf
from tensorflow.python.keras.utils import data_utils

# Tensorflow - Keras
BASE_WEIGTHS_PATH = ('https://storage.googleapis.com/tensorflow/'
                     'keras-applications/densenet/')
DENSENET121_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH + 'densenet121_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET121_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET169_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH + 'densenet169_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET169_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET201_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH + 'densenet201_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET201_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')


def dense_block(x, block, name, k=32):
    for i in range(block):
        x = conv_block(x, k, name=name + '_conv' + str(i + 1))
    return x

def transition_layer(x, name):
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name=name + '_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_relu')(x)
    x = tf.keras.layers.Conv2D(tf.keras.backend.int_shape(x)[-1]//2, (1,1), strides=1, padding='same', name=name + '_conv', use_bias=False)(x)
    x = tf.keras.layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x
    
def conv_block(tensor, growth_rate, name):
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn1')(tensor)
    x = tf.keras.layers.Activation('relu', name=name + '_relu1')(x)
    x = tf.keras.layers.Conv2D(4*growth_rate, 1, use_bias=False, name=name + '_conv1')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn2')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_relu2')(x)
    x = tf.keras.layers.Conv2D(growth_rate, 3, padding='same', use_bias=False, name=name + '_conv2')(x)
    x = tf.keras.layers.Concatenate(axis=bn_axis, name=name + '_concat')([tensor, x])
    return x

def cnn_branch(tensor, name):
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + 'bn1')(tensor)
    x = tf.keras.layers.Activation('relu', name=name + 'relu1')(x)
    x = tf.keras.layers.Conv2D(256, 3, use_bias=False, name=name + 'conv1', padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + 'bn2')(x)
    x = tf.keras.layers.Activation('relu', name=name + 'relu2')(x)
    x = tf.keras.layers.Conv2D(256, 3, use_bias=False, name=name + 'conv2', padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + 'bn3')(x)
    x = tf.keras.layers.Activation('relu', name=name + 'relu3')(x)
    x = tf.keras.layers.Conv2D(256, 3, use_bias=False, name=name + 'conv3', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2), name=name+'_pool1')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + 'bn4')(x)
    x = tf.keras.layers.Activation('relu', name=name + 'relu4')(x)
    x = tf.keras.layers.Conv2D(512, 3, use_bias=False, name=name + 'conv4', padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + 'bn5')(x)
    x = tf.keras.layers.Activation('relu', name=name + 'relu5')(x)
    x = tf.keras.layers.Conv2D(512, 3, use_bias=False, name=name + 'conv5', padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + 'bn6')(x)
    x = tf.keras.layers.Activation('relu', name=name + 'relu6')(x)
    x = tf.keras.layers.Conv2D(512, 3, use_bias=False, name=name + 'conv6', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2), name=name+'_pool2')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2048, activation='relu', name=name+'_fc1')(x)
    x = tf.keras.layers.Dense(2048, activation='relu', name=name+'_fc2')(x)
    x = tf.keras.layers.Dense(1024, activation='relu', name=name+'_fc3')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', name='node'+name)(x)
    
    return x
    
def fc_branch(tensor, name):
    x = tf.keras.layers.Dense(2048, activation='relu', name=name+'_fc1')(tensor)
    x = tf.keras.layers.Dense(2048, activation='relu', name=name+'_fc2')(x)
    x = tf.keras.layers.Dense(1024, activation='relu', name=name+'_fc3')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', name='node'+name)(x)
    return x


def DenseNet(input_shape, classes, act, block=[6,12,48,32]):
    
    input = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=2, padding='same', name='conv1_conv')(input)
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name='conv1_bn')(x)
    x = tf.keras.layers.Activation('relu', name='conv1_relu')(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same', name='pool1')(x)
    
    x = dense_block(x, block[0], name='block1')
    x = transition_layer(x, name='pool2')
    x = dense_block(x, block[1], name='block2')
    x = transition_layer(x, name='pool3')
    x = dense_block(x, block[2], name='block3')
    x = transition_layer(x, name='pool4')
    x = dense_block(x, block[3], name='block4')
    
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name='final_bn')(x)
    x = tf.keras.layers.Activation('relu', name='final_relu')(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D(name='final_avg_pool')(x)
    
    model = tf.keras.models.Model(input, x)
    return model

def ML_DenseNet(img_shape, classes, mltype):
    model = DenseNet(img_shape, classes, act='softmax')
    outputs = []
    if mltype==0:
        end = model.layers[-1].output
        x = tf.keras.layers.Dense(classes, activation=act,name='predictions')(end)
        model = tf.keras.models.Model(model.input, x)
        return model
    elif mltype==1:
        end = model.layers[-229].output
        for i in range(classes):
            x = dense_block(x, 32, name='branch_block')
            x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name='branch_'+str(i+1)+'final_bn')(x)
            x = tf.keras.layers.Activation('relu', name='branch_'+str(i+1)+'final_relu')(x)
            x = tf.keras.layers.GlobalAveragePooling2D(name='branch_'+str(i+1)+'avg_pool')(x)
            out = tf.keras.layers.Dense(1, activation='sigmoid', name='branch_'+str(i+1)+'node')(x)
            outputs.append(out)
        outputs = tf.keras.layers.Concatenate()(outputs)
    elif mltype==2:
        end = model.layers[-229].output
        for i in range(classes):
            out = fc_branch(end, name='branch_'+str(i+1))
            outputs.append(out)
        outputs = tf.keras.layers.Concatenate()(outputs)
    elif mltype==3:
        end = model.layers[-229].output
        for i in range(classes):
            out = cnn_branch(end, name='branch_'+str(i+1))
            outputs.append(out)
        outputs = tf.keras.layers.Concatenate()(outputs)
    elif mltype==4:
        end = model.layers[-2].output
        for i in range(classes):
            out = fc_branch(end, name='branch_'+str(i+1))
            outputs.append(out)
        outputs = tf.keras.layers.Concatenate()(outputs)
    elif mltype==5:
        end = model.layers[-3].output
        for i in range(classes):
            print(i)
            out = cnn_branch(end, name='branch_'+str(i+1))
            outputs.append(out)
        outputs = tf.keras.layers.Concatenate()(outputs)
        
    mlmodel = tf.keras.models.Model(model.input, outputs)
    return mlmodel