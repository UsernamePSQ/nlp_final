from tensorflow.keras.utils import Sequence
from pickle import load
import os


class trainable_emb(Sequence):

    def __init__(self, train=True, up=10, hard_path=False, UMLS=False):
        if hard_path:
            dir_ = '/home/mw/nlp/nlp_final/stat-nlp-book/assignments/2019/final_assignment/problem/Extra_files/resources/'
        else:
            dir_ = os.path.join(os.getcwd(), 'Extra_files/resources')

        if train:
            if not UMLS:
                if up == 10:
                    file = 'tf_train_emb_up10.pkl'
                else:
                    file = 'tf_train_emb.pkl'
            else:
                file = 'tf_train_UMLS.pkl'

        else:
            if not UMLS:
                file = 'tf_dev_emb.pkl'
            else:
                file = 'tf_dev_UMLS.pkl'

        with open(os.path.join(dir_, file), 'rb') as f:
            self.data = load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
