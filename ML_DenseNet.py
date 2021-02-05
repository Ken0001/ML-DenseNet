"""
    ML-DenseNet 
"""
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, GlobalAvgPool2D, BatchNormalization, Concatenate, ReLU
import tensorflow.keras.backend as K

def mldensenet(input_shape, n_classes, mltype=0, finalAct='softmax', f=32):
    # Define repetition
    if mltype == 0 or mltype == 2 or mltype == 3:
        repetitions = 6,12,48,32
    elif mltype == 1 or mltype == 4 or mltype == 5:
        repetitions = 6,12,32

    """ Dense block
        1x1x128 conv + 3x3x32 conv
    """
    def dense_block(tensor, r):
        for i in range(r):
            x = BatchNormalization(epsilon=1.001e-5)(tensor)
            x = ReLU()(x)
            x = Conv2D(f*4, (1, 1), strides=1, padding='same')(x)
            x = BatchNormalization(epsilon=1.001e-5)(x)
            x = ReLU()(x)
            x = Conv2D(f, (3, 3), strides=1, padding='same')(x)
            tensor = Concatenate()([tensor,x])
        return tensor

    """ Transition_layer
        1x1x(input/2) conv + 2x2 pooling
    """
    def transition_layer(x):
        x = Conv2D(K.int_shape(x)[-1]//2, (1,1), strides=1, padding='same')(x)
        x = BatchNormalization(epsilon=1.001e-5)(x)
        x = ReLU()(x)
        x = AveragePooling2D(2, strides=2)(x)
        return x

    """ Dense Block on branch
        Same as default dense block
    """
    def dense_block_branch(x, outputs):
        depth = 16
        for i in range(n_classes):
            d = dense_block(x, depth)
            branch = GlobalAvgPool2D()(d)
            output = Dense(1, activation=finalAct)(branch)
            outputs.append(output)
        outputs = Concatenate()(outputs)
        return outputs

    """ Multilayer Perceptron on branch
        3 fully connected layers
    """
    def fc_branch(x, outputs):
        x = GlobalAvgPool2D()(d)
        for i in range(n_classes):
            x = Dense(1024, activation='relu')(x)
            x = Dense(512, activation='relu')(x)
            x = Dense(256, activation='relu')(x)
            output = Dense(1, activation=finalAct)(x)
            outputs.append(output)
        outputs = Concatenate()(outputs)
        return outputs
    
    """ Convolution Nerual Network on branch
        6 conv layers + 3 fully connected layers
    """
    def cnn_branch(x, outputs):
        for i in range(n_classes):
            v = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            v = Conv2D(256, (3, 3), activation='relu', padding='same')(v)
            v = Conv2D(256, (3, 3), activation='relu', padding='same')(v)
            v = MaxPooling2D((2, 2), strides=(2, 2))(v)
            v = Conv2D(512, (3, 3), activation='relu', padding='same')(v)
            v = Conv2D(512, (3, 3), activation='relu', padding='same')(v)
            v = Conv2D(512, (3, 3), activation='relu', padding='same')(v)
            v = MaxPooling2D((2, 2), strides=(2, 2))(v)
            v = Flatten()(v)
            d = Dense(2048, activation='relu')(v)
            d = Dense(2048, activation='relu')(d)
            d = Dense(1024, activation='relu')(d)
            output = Dense(1, activation=finalAct)(d)
            outputs.append(output)
        outputs = Concatenate()(outputs)
        return outputs

    input = Input(input_shape)
    x = Conv2D(64, (7, 7), strides=2, padding='same')(input)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = ReLU()(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    for r in repetitions:
        d = dense_block(x, r)
        x = transition_layer(d)
    
    outputs = []
    # Original DenseNet
    if mltype==0:
        x = GlobalAvgPool2D()(d)
        outputs = Dense(n_classes, activation=finalAct)(x)

    # ML-DenseNet
    elif mltype==1:
        outputs = dense_block_branch(x, outputs)
    elif mltype==2 or mltype == 4:
        outputs = fc_branch(x, outputs)
    elif mltype==3:
        outputs = cnn_branch(d, outputs)
    elif mltype==5:
        outputs = cnn_branch(x, outputs)
    
    model = Model(input, outputs)
    return model