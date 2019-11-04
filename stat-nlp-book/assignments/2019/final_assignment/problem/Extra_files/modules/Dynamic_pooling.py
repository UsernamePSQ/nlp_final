import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.math import mod, floordiv, greater, logical_and, equal, greater_equal, segment_max


def Dynamic_max_pooling(features, out_len=3):
    def too_short(x: tf.Tensor):
        dyn_shape = tf.shape(x)
        tmp = tf.fill((dyn_shape[0], out_len - dyn_shape[1], dyn_shape[2]), 0.0)
        tmp2 = tf.concat([x, tmp], axis=1)
        return tmp2

    def long_enough(x: tf.Tensor):
        dyn_shape = tf.shape(x)

        n = dyn_shape[1]
        y = floordiv(n, out_len)
        extras = mod(n, out_len)

        current_val = tf.constant(0)
        n_set = tf.constant(0)
        seg = tf.fill((1, n), 0)

        i = tf.constant(0)
        b = lambda i: tf.add(i, 1)

        def body(i, extras, n_set, current_val, seg):
            def case1(i, extras, n_set, current_val: tf.Tensor, seg):
                # current_val += 0
                extras -= 1
                n_set += 1
                seg = tf.sparse.add(seg, tf.SparseTensor(indices=[[0, i]], values=[current_val], dense_shape=[1, n]))
                return extras, n_set, current_val, seg

            def case2(i, extras, n_set, current_val, seg):
                current_val += 1
                n_set -= n_set
                n_set += 1
                seg = tf.sparse.add(seg, tf.SparseTensor(indices=[[0, i]], values=[current_val], dense_shape=[1, n]))
                return extras, n_set, current_val, seg

            def case3(i, extras, n_set, current_val, seg):
                # current_val += 0
                n_set += 1
                seg = tf.sparse.add(seg, tf.SparseTensor(indices=[[0, i]], values=[current_val], dense_shape=[1, n]))
                return extras, n_set, current_val, seg

            extras, n_set, current_val, seg = tf.cond(logical_and(greater(extras, 0), equal(n_set, y)),
                                                      lambda: case1(i, extras, n_set, current_val, seg),
                                                      lambda: tf.cond(greater_equal(n_set, y),
                                                                      lambda: case2(i, extras, n_set, current_val, seg),
                                                                      lambda: case3(i, extras, n_set, current_val,
                                                                                    seg)))
            return b(i), extras, n_set, current_val, seg

        c = lambda i, extras, n_set, current_val, seg: tf.less(i, n)

        r, extras, n_set, current_val, seg = tf.while_loop(c, body, [i, extras, n_set, current_val, seg])
        seg = tf.squeeze(seg)

        return tf.expand_dims(
            tf.transpose(tf.map_fn(lambda a: segment_max(a, seg), tf.transpose(tf.squeeze(x, [0]), [1, 0])), [1, 0]), 0)
        # return tf.transpose(tf.squeeze(x), [1, 0])
        # return tf.shape(tf.squeeze(x))
        # return tf.expand_dims(tf.map_fn(lambda a: tf.shape(tf.squeeze(a)), tf.transpose(tf.squeeze(x), [1, 0])),0)
        # return tf.expand_dims(tf.transpose(tf.squeeze(x), [1, 0]),0)

    def f(x: tf.Tensor):
        br = tf.constant(out_len)
        # print(br.shape)
        # tmp = tf.shape(x)
        # low = tf.constant([[0.0, 2]])
        # high = tf.constant([2.0])

        ret = tf.cond(tf.less(tf.shape(x)[1], br), lambda: too_short(x), lambda: long_enough(x))
        # ret = tf.cond(tf.less(tf.shape(x)[1], br), lambda:low, lambda: low)

        return tf.reshape(ret, [1, out_len, features])
        # return ret
        # return long_enough(x)

    return Lambda(f)


if __name__ == '__main__':
    import numpy as np
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Embedding, Input, Concatenate, Conv1D, Lambda, Flatten, Dropout

    n = 1000
    xn = 5
    train_x = [np.expand_dims(np.asarray([[float(i), i + 1]] * xn), axis=0) for i in range(n)]
    train_x2 = [np.expand_dims(np.asarray([[float(i), i + 1]] * (xn + i % round(xn / 2))), axis=0) for i in range(n)]
    train_y = [np.expand_dims(np.asarray(int(i < xn // 2)), axis=0) for i in range(n)]


    def gen():
        while True:
            for x,y  in zip(train_x2, train_y):
                yield [x,x], y


    xin = Input(batch_shape=(1, None, 2))
    xin2 = Input(batch_shape=(1, None, 2))
    x = Conv1D(filters=2, kernel_size=3, strides=1)(xin)
    x = Dynamic_max_pooling(2, 10)(x)
    x = Flatten()(x)
    x = Dense(10)(x)
    x = Dense(2)(x)
    model = Model(inputs=[xin, xin2], outputs=x)
    model.summary()
    # tmp = model.predict(np.expand_dims(train_x2[-1], axis=0))
    tmp = model.predict([train_x2[0], train_x2[0]])
    print(tmp)
    a = np.reshape(range(2 * 3), (1, 3, 2))
    # a = np.reshape(range(2 * 2), (1, 2, 2))
    tmp = model.predict(a)
    print(tmp)

    # model.predict(np.expand_dims(list(range(5)), axis=0))

    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    # model.compile(loss='mean_squared_error',
    #               optimizer='adam',
    #               metrics=['sparse_categorical_crossentropy'])

    e = 1
    model.fit_generator(gen(), steps_per_epoch=n, epochs=e)
    model.fit(np.squeeze(np.asarray(train_x)), np.squeeze(np.asarray(train_y)))
    tmp = model.predict(np.squeeze(np.asarray(train_x)))
    print(tmp)
