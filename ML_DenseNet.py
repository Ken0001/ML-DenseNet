"""
ML-DenseNet
"""
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, BatchNormalization, Concatenate, Add, ReLU, Activation, Reshape, Permute, multiply
import tensorflow.keras.backend as K
"""
DenseNet_ml1 -> Multiple branch based on DenseNet
DenseNet_ml2 -> 4 dense blocks + fc
DenseNet_ml3 -> 4 dense blocks + cnn
DenseNet_ml4 -> 3 dense blocks + fc
DenseNet_ml5 -> 3 dense blocks + cnn
"""
def mldensenet(img_shape, n_classes, mltype=0, finalAct='softmax', f=32):
    """
    if mltype == 0:
        repetitions = 6,12,24,16
    elif mltype == 2 or mltype == 3:
        repetitions = 6,12,24,16
    elif mltype == 1 or mltype == 4 or mltype == 5:
        repetitions = 6,12,24
    """ 
    if mltype == 0:
        repetitions = 6,12,48,32
    elif mltype == 2 or mltype == 3:
        repetitions = 6,12,48,32
    elif mltype == 1 or mltype == 4 or mltype == 5:
        repetitions = 6,12,32

    def bn_rl_conv(x, f, k=1, s=1, p='same'):
        x = BatchNormalization(epsilon=1.001e-5)(x)
        x = ReLU()(x)
        x = Conv2D(f, k, strides=s, padding=p)(x)
        return x
    
    def dense_block(tensor, r):
        for _ in range(r):
            x = bn_rl_conv(tensor, 4*f)
            x = bn_rl_conv(x, f, 3)
            x = squeeze_excite_block(x)
            tensor = Concatenate()([tensor, x])
        return tensor
    
    def transition_block(x):
        x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
        #x = Dropout(0.2)(x)
        x = AvgPool2D(2, strides=2, padding='same')(x)
        return x

    def squeeze_excite_block(tensor, ratio=16):
        init = tensor
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        filters = init.shape[channel_axis]
        se_shape = (1, 1, filters)

        se = GlobalAvgPool2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

        if K.image_data_format() == 'channels_first':
            se = Permute((3, 1, 2))(se)

        x = multiply([init, se])
        return x
    
    input = Input(img_shape)
    
    x = Conv2D(64, 7, strides=2, padding='same')(input)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = ReLU()(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    
    for r in repetitions:
        d = dense_block(x, r)
        #d = squeeze_excite_block(d)
        x = transition_block(d)
    

    if mltype==0:
        x = GlobalAvgPool2D()(d)
        output = Dense(n_classes, activation=finalAct)(x)
        model = Model(input, output)
        return model
    elif mltype==1:
        outputs = []
        mt_block = 16
        for i in range(n_classes):
            print("class ", i)
            d = dense_block(x, mt_block)
            branch = GlobalAvgPool2D()(d)
            output = Dense(1, activation=finalAct)(branch)
            outputs.append(output)
        outputs = Concatenate()(outputs)
        model = Model(input, outputs)
        return model
    elif mltype==2:
        x = GlobalAvgPool2D()(d)
        outputs = []
        for i in range(n_classes):
            print("\rclass ", i, end="")
            x = Dense(1024, activation='relu')(x)
            x = Dense(512, activation='relu')(x)
            x = Dense(256, activation='relu')(x)
            output = Dense(1, activation=finalAct)(x)
            outputs.append(output)
        outputs = Concatenate()(outputs)
        model = Model(input, outputs)
        return model
    elif mltype==3:
        a = transition_block(d)
        outputs = []
        for i in range(n_classes):
            print("\rclass ", i, end="")
            v = Conv2D(256, (3, 3), activation='relu', padding='same')(a)
            v = Conv2D(256, (3, 3), activation='relu', padding='same')(v)
            v = Conv2D(256, (3, 3), activation='relu', padding='same')(v)
            v = MaxPooling2D((2, 2), strides=(2, 2))(v)
            v = Conv2D(512, (3, 3), activation='relu', padding='same')(v)
            v = Conv2D(512, (3, 3), activation='relu', padding='same')(v)
            v = Conv2D(512, (3, 3), activation='relu', padding='same')(v)
            v = MaxPooling2D((2, 2), strides=(2, 2))(v)
            v = Flatten()(v)
            d = Dense(4096, activation='relu')(v)
            d = Dense(4096, activation='relu')(d)
            d = Dense(1024, activation='relu')(d)
            output = Dense(1, activation=finalAct)(d)
            outputs.append(output)
        outputs = Concatenate()(outputs)
        model = Model(input, outputs)
        return model
    elif mltype==4:
        x = GlobalAvgPool2D()(d)
        outputs = []
        for i in range(n_classes):
            print("\rclass ", i, end="")
            x = Dense(1024, activation='relu')(x)
            x = Dense(512, activation='relu')(x)
            x = Dense(256, activation='relu')(x)
            output = Dense(1, activation=finalAct)(x)
            outputs.append(output)
        outputs = Concatenate()(outputs)
        model = Model(input, outputs)
        return model
    elif mltype==5:
        a = transition_block(d)
        outputs = []
        for i in range(n_classes):
            print("\rclass ", i, end="")
            v = Conv2D(256, (3, 3), activation='relu', padding='same')(a)
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
        model = Model(input, outputs)
        return model