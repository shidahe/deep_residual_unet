from keras.models import *
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Activation, add, concatenate

def res_block(x, nb_filters, strides):
    net = {}
    net['input'] = x

    shortcut = {}
    shortcut['conv'] = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(net['input'])
    shortcut['batch_norm'] = BatchNormalization()(shortcut['conv'])


    net['batch_norm_1'] = BatchNormalization()(net['input'])
    net['activation_1'] = Activation(activation='relu')(net['batch_norm_1'])
    net['conv_1'] = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(net['activation_1'])

    net['concat'] = concatenate([net['conv_1'], shortcut['batch_norm']])

    net['batch_norm_2'] = BatchNormalization()(net['concat'])
    net['activation_2'] = Activation(activation='relu')(net['batch_norm_2'])
    net['conv_2'] = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(net['activation_2'])

    res_path = concatenate([shortcut['batch_norm'], net['conv_2']])
    return res_path


def encoder(x):
    to_decoder = []

    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='relu')(main_path)

    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)

    shortcut = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1))(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = concatenate([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    return to_decoder


def decoder(x, from_encoder):
    main_path = UpSampling2D(size=(2, 2))(x)
    main_path = concatenate([main_path, from_encoder[2]], axis=3)
    main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=3)
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=3)
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])

    return main_path


def build(input_shape):
    inputs = Input(shape=input_shape)

    to_decoder = encoder(inputs)

    path = res_block(to_decoder[2], [512, 512], [(2, 2), (1, 1)])

    path = decoder(path, from_encoder=to_decoder)

    path = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(path)

    return Model(input=inputs, output=path)