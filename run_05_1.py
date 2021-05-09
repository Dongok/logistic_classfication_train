import tensorflow as tf


def train():
    x_data = [[1, 2],
              [2, 3],
              [3, 1],
              [4, 3],
              [5, 3],
              [6, 2]]
    y_data = [[0],
              [0],
              [0],
              [1],
              [1],
              [1]]

    tf.model = tf.keras.Sequential()
    tf.model.add(tf.keras.layers.Dense(units=1, input_dim=2))
    # use sigmoid activation for 0~1 problem
    tf.model.add(tf.keras.layers.Activation('sigmoid'))

    ''' 
    better result with loss function == 'binary_crossentropy', try 'mse' for yourself
    adding accuracy metric to get accuracy report during training
    '''
    tf.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])
    tf.model.summary()

    history = tf.model.fit(x_data, y_data, epochs=50)

    # Accuracy report
    print("Accuracy: ", history.history['accuracy'][-1])
    print(history)



if __name__ == '__main__':
    train()
