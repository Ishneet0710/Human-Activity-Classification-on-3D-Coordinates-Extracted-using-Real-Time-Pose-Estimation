import os
os.environ['TF_cpp_MIN_LEVEL'] =  '2'

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


class CsvDataset:

    def __init__(self, file):
        self.dataframe = pd.read_csv(file)
        self.val_df = None
        self.train_df = None
        self.val_ds = None
        self.train_ds = None

    def csv_preprocessing(self):
        ''''Remove points 1 to points 11 of columns.'''

        df = self.dataframe.copy()

        columns_removed = [
            'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11',
            'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11',
            'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11',
            'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11',
            'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21',
            'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31',
            'v32', 'v33',
        ]

        df = df.drop(columns_removed, axis = 'columns')

        return df
    
    def df_to_datasets(self, dataframe, target):
        ''''Input pandas dataframe to get specific features dataset of datasets.'''

        self.val_df = dataframe.sample(frac=0.3, random_state=42)
        self.train_df = dataframe.drop(self.val_df.index)

        train_df = self.train_df.copy()
        val_df = self.val_df.copy()

        train_labels = train_df.pop(target)
        val_labels = val_df.pop(target)
        
        # tf.data.Dataset.from_tensor_slices(): 
        self.train_ds = tf.data.Dataset.from_tensor_slices((dict(train_df), train_labels))
        self.val_ds = tf.data.Dataset.from_tensor_slices((dict(val_df), val_labels))


        return self.train_ds, self.val_ds


class EncodeFeatures:

    def __init__(self):
        self.feature_ds = None
            
    def numerical_feature(self, feature, name, dataset):
        # Create a Normalization layer for our feature
        normalizer = tf.keras.layers.Normalization()

        # Prepare a Dataset that only yields our feature
        self.feature_ds = dataset.map(lambda x, y: x[name])
        self.feature_ds = self.feature_ds.map(lambda x: tf.expand_dims(x, -1))

        # Learn the statistics of the data
        normalizer.adapt(self.feature_ds)

        # Normalize the input feature
        encoded_feature = normalizer(feature)
        return encoded_feature


def point():
    x = [
        'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22',
        'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'x31', 'x32', 'x33',
    ]
    y = [
        'y12', 'y13', 'y14', 'y15', 'y16', 'y17', 'y18', 'y19', 'y20', 'y21', 'y22',
        'y23', 'y24', 'y25', 'y26', 'y27', 'y28', 'y29', 'y30', 'y31', 'y32', 'y33',
    ]
    z = [
        'z12', 'z13', 'z14', 'z15', 'z16', 'z17', 'z18', 'z19', 'z20', 'z21', 'z22',
        'z23', 'z24', 'z25', 'z26', 'z27', 'z28', 'z29', 'z30', 'z31', 'z32', 'z33',
    ]
    coords = [x, y, z]
    return coords


def input_features():

    coords = point()
    # point 12 ~ 33
    # Numerical features
    x12 = keras.Input(shape=(1,), name=coords[0][0])
    x13 = keras.Input(shape=(1,), name=coords[0][1])
    x14 = keras.Input(shape=(1,), name=coords[0][2])
    x15 = keras.Input(shape=(1,), name=coords[0][3])
    x16 = keras.Input(shape=(1,), name=coords[0][4])
    x17 = keras.Input(shape=(1,), name=coords[0][5]) 
    x18 = keras.Input(shape=(1,), name=coords[0][6])
    x19 = keras.Input(shape=(1,), name=coords[0][7])
    x20 = keras.Input(shape=(1,), name=coords[0][8])
    x21 = keras.Input(shape=(1,), name=coords[0][9])
    x22 = keras.Input(shape=(1,), name=coords[0][10])
    x23 = keras.Input(shape=(1,), name=coords[0][11])
    x24 = keras.Input(shape=(1,), name=coords[0][12])
    x25 = keras.Input(shape=(1,), name=coords[0][13])
    x26 = keras.Input(shape=(1,), name=coords[0][14])
    x27 = keras.Input(shape=(1,), name=coords[0][15])
    x28 = keras.Input(shape=(1,), name=coords[0][16])
    x29 = keras.Input(shape=(1,), name=coords[0][17])
    x30 = keras.Input(shape=(1,), name=coords[0][18])
    x31 = keras.Input(shape=(1,), name=coords[0][19])
    x32 = keras.Input(shape=(1,), name=coords[0][20])
    x33 = keras.Input(shape=(1,), name=coords[0][21])

    y12 = keras.Input(shape=(1,), name=coords[1][0])
    y13 = keras.Input(shape=(1,), name=coords[1][1])
    y14 = keras.Input(shape=(1,), name=coords[1][2])
    y15 = keras.Input(shape=(1,), name=coords[1][3])
    y16 = keras.Input(shape=(1,), name=coords[1][4])
    y17 = keras.Input(shape=(1,), name=coords[1][5]) 
    y18 = keras.Input(shape=(1,), name=coords[1][6])
    y19 = keras.Input(shape=(1,), name=coords[1][7])
    y20 = keras.Input(shape=(1,), name=coords[1][8])
    y21 = keras.Input(shape=(1,), name=coords[1][9])
    y22 = keras.Input(shape=(1,), name=coords[1][10])
    y23 = keras.Input(shape=(1,), name=coords[1][11])
    y24 = keras.Input(shape=(1,), name=coords[1][12])
    y25 = keras.Input(shape=(1,), name=coords[1][13])
    y26 = keras.Input(shape=(1,), name=coords[1][14])
    y27 = keras.Input(shape=(1,), name=coords[1][15])
    y28 = keras.Input(shape=(1,), name=coords[1][16])
    y29 = keras.Input(shape=(1,), name=coords[1][17])
    y30 = keras.Input(shape=(1,), name=coords[1][18])
    y31 = keras.Input(shape=(1,), name=coords[1][19])
    y32 = keras.Input(shape=(1,), name=coords[1][20])
    y33 = keras.Input(shape=(1,), name=coords[1][21])

    z12 = keras.Input(shape=(1,), name=coords[2][0])
    z13 = keras.Input(shape=(1,), name=coords[2][1])
    z14 = keras.Input(shape=(1,), name=coords[2][2])
    z15 = keras.Input(shape=(1,), name=coords[2][3])
    z16 = keras.Input(shape=(1,), name=coords[2][4])
    z17 = keras.Input(shape=(1,), name=coords[2][5]) 
    z18 = keras.Input(shape=(1,), name=coords[2][6])
    z19 = keras.Input(shape=(1,), name=coords[2][7])
    z20 = keras.Input(shape=(1,), name=coords[2][8])
    z21 = keras.Input(shape=(1,), name=coords[2][9])
    z22 = keras.Input(shape=(1,), name=coords[2][10])
    z23 = keras.Input(shape=(1,), name=coords[2][11])
    z24 = keras.Input(shape=(1,), name=coords[2][12])
    z25 = keras.Input(shape=(1,), name=coords[2][13])
    z26 = keras.Input(shape=(1,), name=coords[2][14])
    z27 = keras.Input(shape=(1,), name=coords[2][15])
    z28 = keras.Input(shape=(1,), name=coords[2][16])
    z29 = keras.Input(shape=(1,), name=coords[2][17])
    z30 = keras.Input(shape=(1,), name=coords[2][18])
    z31 = keras.Input(shape=(1,), name=coords[2][19])
    z32 = keras.Input(shape=(1,), name=coords[2][20])
    z33 = keras.Input(shape=(1,), name=coords[2][21])


    all_inputs = [
        x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, 
        x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33,

        y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, 
        y23, y24, y25, y26, y27, y28, y29, y30, y31, y32, y33,

        z12, z13, z14, z15, z16, z17, z18, z19, z20, z21, z22, 
        z23, z24, z25, z26, z27, z28, z29, z30, z31, z32, z33,
    ]

    return all_inputs


def specify_encoded_features(train_ds, all_inputs):

    encoded = EncodeFeatures()
    coords_p = point()

    # point 12 ~ 33
    # Numerical features
    x12 = encoded.numerical_feature(all_inputs[0], coords_p[0][0], train_ds)
    x13 = encoded.numerical_feature(all_inputs[1], coords_p[0][1], train_ds)
    x14 = encoded.numerical_feature(all_inputs[2], coords_p[0][2], train_ds)
    x15 = encoded.numerical_feature(all_inputs[3], coords_p[0][3], train_ds)
    x16 = encoded.numerical_feature(all_inputs[4], coords_p[0][4], train_ds)
    x17 = encoded.numerical_feature(all_inputs[5], coords_p[0][5], train_ds)
    x18 = encoded.numerical_feature(all_inputs[6], coords_p[0][6], train_ds)
    x19 = encoded.numerical_feature(all_inputs[7], coords_p[0][7], train_ds)
    x20 = encoded.numerical_feature(all_inputs[8], coords_p[0][8], train_ds)
    x21 = encoded.numerical_feature(all_inputs[9], coords_p[0][9], train_ds)
    x22 = encoded.numerical_feature(all_inputs[10], coords_p[0][10], train_ds)
    x23 = encoded.numerical_feature(all_inputs[11], coords_p[0][11], train_ds)
    x24 = encoded.numerical_feature(all_inputs[12], coords_p[0][12], train_ds)
    x25 = encoded.numerical_feature(all_inputs[13], coords_p[0][13], train_ds)
    x26 = encoded.numerical_feature(all_inputs[14], coords_p[0][14], train_ds)
    x27 = encoded.numerical_feature(all_inputs[15], coords_p[0][15], train_ds)
    x28 = encoded.numerical_feature(all_inputs[16], coords_p[0][16], train_ds)
    x29 = encoded.numerical_feature(all_inputs[17], coords_p[0][17], train_ds)
    x30 = encoded.numerical_feature(all_inputs[18], coords_p[0][18], train_ds)
    x31 = encoded.numerical_feature(all_inputs[19], coords_p[0][19], train_ds)
    x32 = encoded.numerical_feature(all_inputs[20], coords_p[0][20], train_ds)
    x33 = encoded.numerical_feature(all_inputs[21], coords_p[0][21], train_ds)

    y12 = encoded.numerical_feature(all_inputs[22], coords_p[1][0], train_ds)
    y13 = encoded.numerical_feature(all_inputs[23], coords_p[1][1], train_ds)
    y14 = encoded.numerical_feature(all_inputs[24], coords_p[1][2], train_ds)
    y15 = encoded.numerical_feature(all_inputs[25], coords_p[1][3], train_ds)
    y16 = encoded.numerical_feature(all_inputs[26], coords_p[1][4], train_ds)
    y17 = encoded.numerical_feature(all_inputs[27], coords_p[1][5], train_ds)
    y18 = encoded.numerical_feature(all_inputs[28], coords_p[1][6], train_ds)
    y19 = encoded.numerical_feature(all_inputs[29], coords_p[1][7], train_ds)
    y20 = encoded.numerical_feature(all_inputs[30], coords_p[1][8], train_ds)
    y21 = encoded.numerical_feature(all_inputs[31], coords_p[1][9], train_ds)
    y22 = encoded.numerical_feature(all_inputs[32], coords_p[1][10], train_ds)
    y23 = encoded.numerical_feature(all_inputs[33], coords_p[1][11], train_ds)
    y24 = encoded.numerical_feature(all_inputs[34], coords_p[1][12], train_ds)
    y25 = encoded.numerical_feature(all_inputs[35], coords_p[1][13], train_ds)
    y26 = encoded.numerical_feature(all_inputs[36], coords_p[1][14], train_ds)
    y27 = encoded.numerical_feature(all_inputs[37], coords_p[1][15], train_ds)
    y28 = encoded.numerical_feature(all_inputs[38], coords_p[1][16], train_ds)
    y29 = encoded.numerical_feature(all_inputs[39], coords_p[1][17], train_ds)
    y30 = encoded.numerical_feature(all_inputs[40], coords_p[1][18], train_ds)
    y31 = encoded.numerical_feature(all_inputs[41], coords_p[1][19], train_ds)
    y32 = encoded.numerical_feature(all_inputs[42], coords_p[1][20], train_ds)
    y33 = encoded.numerical_feature(all_inputs[43], coords_p[1][21], train_ds)

    z12 = encoded.numerical_feature(all_inputs[44], coords_p[2][0], train_ds)
    z13 = encoded.numerical_feature(all_inputs[45], coords_p[2][1], train_ds)
    z14 = encoded.numerical_feature(all_inputs[46], coords_p[2][2], train_ds)
    z15 = encoded.numerical_feature(all_inputs[47], coords_p[2][3], train_ds)
    z16 = encoded.numerical_feature(all_inputs[48], coords_p[2][4], train_ds)
    z17 = encoded.numerical_feature(all_inputs[49], coords_p[2][5], train_ds)
    z18 = encoded.numerical_feature(all_inputs[50], coords_p[2][6], train_ds)
    z19 = encoded.numerical_feature(all_inputs[51], coords_p[2][7], train_ds)
    z20 = encoded.numerical_feature(all_inputs[52], coords_p[2][8], train_ds)
    z21 = encoded.numerical_feature(all_inputs[53], coords_p[2][9], train_ds)
    z22 = encoded.numerical_feature(all_inputs[54], coords_p[2][10], train_ds)
    z23 = encoded.numerical_feature(all_inputs[55], coords_p[2][11], train_ds)
    z24 = encoded.numerical_feature(all_inputs[56], coords_p[2][12], train_ds)
    z25 = encoded.numerical_feature(all_inputs[57], coords_p[2][13], train_ds)
    z26 = encoded.numerical_feature(all_inputs[58], coords_p[2][14], train_ds)
    z27 = encoded.numerical_feature(all_inputs[59], coords_p[2][15], train_ds)
    z28 = encoded.numerical_feature(all_inputs[60], coords_p[2][16], train_ds)
    z29 = encoded.numerical_feature(all_inputs[61], coords_p[2][17], train_ds)
    z30 = encoded.numerical_feature(all_inputs[62], coords_p[2][18], train_ds)
    z31 = encoded.numerical_feature(all_inputs[63], coords_p[2][19], train_ds)
    z32 = encoded.numerical_feature(all_inputs[64], coords_p[2][20], train_ds)
    z33 = encoded.numerical_feature(all_inputs[65], coords_p[2][21], train_ds)

    all_features = layers.concatenate(
        [
            x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, 
            x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33,

            y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22,
            y23, y24, y25, y26, y27, y28, y29, y30, y31, y32, y33,

            z12, z13, z14, z15, z16, z17, z18, z19, z20, z21, z22, 
            z23, z24, z25, z26, z27, z28, z29, z30, z31, z32, z33,

        ]
    )
    return all_features


if __name__ == '__main__':

   

    dataset_csv_file = './datasets/coords_dataset.csv'
    target_value = 'class'

    # Data preprocessed and creat datasets.
    pose_datasets = CsvDataset(file=dataset_csv_file)
    df_pose = pose_datasets.csv_preprocessing()
    train_ds, val_ds = pose_datasets.df_to_datasets(dataframe=df_pose, target=target_value)


    train_ds = train_ds.batch(32)
    val_ds = val_ds.batch(32)

    # model build.
    all_inputs = input_features()
    all_features = specify_encoded_features(train_ds, all_inputs)
    x = layers.Dense(32, activation='relu')(all_features)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(3, activation='softmax')(x) 
    model = keras.Model(all_inputs, output)


    model.compile(optimizer='sgd', 
                  loss=keras.losses.SparseCategoricalCrossentropy(), 
                  metrics=['accuracy'])

    # Model train.
    history = model.fit(x=train_ds, epochs=10, verbose=2, validation_data=val_ds)



    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

    print('TFLite Model save done!')