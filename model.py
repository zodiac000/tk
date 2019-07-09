from tensorflow.keras import datasets, layers, models

shape = 224


def create_model():
    model = models.Sequential()
    # model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), input_shape=(shape, shape, 1)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.5))

    # model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.5))

    # model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(32, activation='relu'))
    # model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(2, activation='sigmoid'))

    model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

    return model

# def create_model2():
    # X = tf.placeholder(tf.float32, shape=(None, n_inputs, n_inputs, 1), name = "X")
    # y = tf.placeholder(tf.float32, shape=(None, 3, name="y")

    # with tf.name_scope("conv1"):
        # conv1 = tf.compat.v1.layers.conv2d(X, 32, 3, activation=tf.nn.relu)
    # with tf.name_scope("pool1"):
        # pool1 = tf.compat.v1.layers.max_pooling2d(conv1, 2, 2) 
    # with tf.name_scope("conv2"):
        # conv2 = tf.compat.v1.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu)
    # with tf.name_scope("pool2"):
        # pool2 = tf.compat.v1.layers.max_pooling2d(conv2, 2, 2) 
    # with tf.name_scope("conv3"):
        # conv3 = tf.compat.v1.layers.conv2d(pool2, 64, 3, activation=tf.nn.relu)
        # conv3_flat = tf.compat.v1.layers.flatten(pool1)
    # with tf.name_scope("fc1"):
        # fc1 = tf.compat.v1.layers.dense(conv3_flat, 64, activation=tf.nn.relu, name="fc1")
    # with tf.name_scope("fc2"):
        # logits = tf.compat.v1.layers.dense(fc1, 3, name="output")


if __name__ == "__main__":
    model = create_model()

