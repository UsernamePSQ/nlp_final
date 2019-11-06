import numpy as np


def pre_embedded(enc, xs, ys=None, once=False):
    """
    Catch all function:
        makes data trainable/predictable by yielding batches of single observations, i.e.
        compatable with the input of the NN
    """

    def to_col_list(sequence):
        return [np.expand_dims(x, 0) if i in [3, 4] else np.expand_dims(np.expand_dims(x, -1), 0) for i, x in
                enumerate(list(np.swapaxes(sequence, 1, 0)))]

    def onehot(y):
        return enc.transform(np.array(y).reshape(-1, 1)).todense()

    if ys is not None:
        while True:
            for x, y in zip(xs, ys):
                yield to_col_list(x), onehot(y)
            if once:
                break
    else:
        while True:
            for x in xs:
                yield to_col_list(x)
            if once:
                break
