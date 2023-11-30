
import tensorflow as tf
from keras.layers import Input, Dense,Flatten
# from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate, BatchNormalization, Dense, add
from keras.models import Model, model_from_json



def SEBlock(inputs,reduction=8,if_train=True):
    x = tf.keras.layers.GlobalAveragePooling1D()(inputs)
    x = tf.keras.layers.Dense(int(x.shape[-1]//reduction),use_bias=False,activation=tf.keras.activations.relu,trainable=if_train)(x)
    x = tf.keras.layers.Dense(int(inputs.shape[-1]),use_bias=False,activation=tf.keras.activations.hard_sigmoid,trainable=if_train)(x)
    return tf.keras.layers.Multiply()([inputs,x])

def SENet(x):
    inputs = Input((500, 1))
    conv1 = Conv1D(x, 2, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv1D(x, 2, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)


    conv2 = Conv1D(x * 4, 2, activation='relu', padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv1D(x * 4, 2, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)


    conv3 = Conv1D(x * 8, 3, activation='relu', padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv1D(x * 8, 3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(x * 4, 3, activation='relu', padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv1D(x * 4, 3, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling1D(pool_size=4)(conv4)

    conv5 = Conv1D(x * 4, 2, activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv1D(x * 4, 2, activation='relu', padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)


    conv4_se = SEBlock(conv4)

    up6 = concatenate([conv5, conv4_se], axis=2)

    conv6 = Conv1D(x * 8, 3, activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv1D(x * 8, 3, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    pool6 = MaxPooling1D(pool_size=4)(conv6)

    conv3_se = SEBlock(conv3)
    up7 = concatenate([conv6, conv3_se], axis=2)  #
    conv7 = Conv1D(x * 4, 3, activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv1D(x * 4, 3, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    pool7 = MaxPooling1D(pool_size=4)(conv4)

    conv2_se = SEBlock(conv2)
    up8 = concatenate([UpSampling1D(size=2)(conv7), conv2_se], axis=2)
    conv8 = Conv1D(x, 3, activation='relu', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv1D(x, 3, activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)


    conv1_se = SEBlock(conv1)
    up9 = concatenate([conv8, conv1_se], axis=2)
    conv9 = Conv1D(2, 2, activation='relu', padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv1D(2, 2, activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Flatten()(conv9)

    conv10 = Dense(250)(conv9)

    out_reg1 = Dense(1, activation='linear')(conv9)
    out_reg2 = Dense(1, activation='linear')(conv9)


    model = Model(inputs=[inputs], outputs=[conv10, out_reg1, out_reg2])

    return model

