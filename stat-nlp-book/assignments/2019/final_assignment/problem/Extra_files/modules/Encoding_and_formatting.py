import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import Sequence


## MW - QUICK MODEL - NEEDED GLOBAL VARIABLES 

# DON'T TOUCH, WILL BE MOVED / CLEANED

enc = OneHotEncoder()
### NfS
# enc.fit(np.asarray(data_m_XY['data_Y']).reshape(-1,1))
enc.fit(np.asarray(['Hyponym', 'Hypernym', 'NONE', 'Synonym']).reshape(-1,1))
###

def pre_embedded(enc, xs, ys=None, once=False, UMLS=None):
    """
    Catch all function:
        makes data trainable/predictable by yielding batches of single observations, i.e.
        compatable with the input of the NN
    """

    def to_col_list(sequence):
        return [np.expand_dims(x, 0) if i in [3, 4] else np.expand_dims(np.expand_dims(x, -1), 0) if i in [1,2] else np.expand_dims(np.stack(x), 0) for i, x in
                    enumerate(list(np.swapaxes(sequence, 1, 0)))]
    def onehot(y):
        return enc.transform(np.array(y).reshape(-1, 1)).todense()

    if ys is not None and UMLS is None:
        while True:
            for x, y in zip(xs, ys):
                yield to_col_list(x), onehot(y)
            if once:
                break
                
    elif ys is None and UMLS is None:
        while True:
            for x in xs:
                yield to_col_list(x)
            if once:
                break

    elif ys is None and UMLS is not None:
        while True:
            tmp = list(sorted(set(UMLS)))  # convert to index
            for x, u in zip(xs, UMLS):
                out_x = to_col_list(x)
                out_x.append(np.expand_dims(np.asarray([tmp.index(u)]),0))
                yield out_x
            if once:
                break

    else:
        while True:
            tmp = list(sorted(set(UMLS)))  # convert to index
            for x, y, u in zip(xs, ys, UMLS):
                out_x = to_col_list(x)
                out_x.append(np.expand_dims(np.asarray([tmp.index(u)]),0))
                yield out_x, onehot(y)
            if once:
                break


class keras_seq(Sequence):

    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def to_tf_format(enc, xs, ys, UMLS=None):
    return keras_seq(list(pre_embedded(enc, xs, ys, True, UMLS)))
