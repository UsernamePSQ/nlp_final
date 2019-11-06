from Extra_files.modules.Dynamic_pooling import Dynamic_max_pooling
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input, Concatenate, Conv1D, Flatten, Dropout, LeakyReLU
from tensorflow.keras.losses import categorical_crossentropy  # , sparse_categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

import os
from datetime import datetime


class fasttext_cnn_model:
    def __init__(self, filters=200, dyn_out_len=100, kernel_size=3, name: str = ''):
        ## PARAMETERS

        # Embeddings
        # vocab_size = 4092
        entity_vocab_size = 20
        pos_vocab_size = 20

        # word_emb_dim = 50
        entity_emb_dim = 10
        pos_emb_dim = 10

        # Conv1D
        # filters = 20
        conv_kernel_size = kernel_size
        conv_strides = 1

        # Pooling
        # dyn_out_len = 10

        self.parameters = (name, entity_vocab_size, pos_vocab_size, entity_emb_dim, pos_emb_dim, filters,
                           conv_kernel_size, conv_strides, kernel_size, dyn_out_len)
        ## LAYERS

        # Input
        tokens = Input(batch_shape=(1, None, 51))
        relpos1 = Input(batch_shape=(1, None, 1))
        relpos2 = Input(batch_shape=(1, None, 1))
        entity = Input(batch_shape=(1, None))
        pos = Input(batch_shape=(1, None))
        lemma = Input(batch_shape=(1, None, 51))

        # Embedding
        # word_emb = Embedding(vocab_size, word_emb_dim)(tokens)
        entity_emb = Embedding(entity_vocab_size, entity_emb_dim)(entity)
        pos_emb = Embedding(pos_vocab_size, pos_emb_dim)(pos)
        # lemma_emb = Embedding(vocab_size, word_emb_dim)(lemma)

        # Base-model
        #     x = concatenate([word_emb, relpos1, relpos2, entity_emb, pos_emb, lemma_emb], axis=2)
        x = Concatenate(axis=2)([tokens, relpos1, relpos2, entity_emb, pos_emb, lemma])
        x = Dropout(0.5)(x)
        x = Conv1D(filters=filters, kernel_size=conv_kernel_size, strides=conv_strides, padding='same', activation='relu')(x)
        # x = LeakyReLU(0.3)(x)
        x = Dynamic_max_pooling(filters, dyn_out_len)(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(4, activation='softmax')(x)  # 4 == n_classes in data (i.e. NOT a parameter)

        ## Create model
        model = Model(inputs=[tokens, relpos1, relpos2, entity, pos, lemma], outputs=x)
        #     model.summary()

        model.compile(loss=categorical_crossentropy,  # for onehot, for idx: sparse_categorical_crossentropy
                      optimizer=Adam(learning_rate=0.001, epsilon=1e-7, amsgrad=False),
                      metrics=['accuracy'])

        self.model = model

        self.base_dir = os.path.join(os.getcwd(), 'Extra_files/resources/',
                                     self.__class__.__name__ + '_' + str(filters) + '_' + str(dyn_out_len) + '_' + str(
                                         kernel_size))

        os.makedirs(self.base_dir, exist_ok=True)
        dir_ = self.base_dir
        names = [x for x in os.listdir(dir_)]
        names.sort(key=lambda x: int(x.split('_')[0]))
        h = hash(self.parameters)
        dir_set = False
        num = 0
        for x in names:
            num, hash_ = x.split('_')
            if h == int(hash_):
                dir_ = os.path.join(dir_, x)
                dir_set = True
                break

        if not dir_set:
            dir_ = os.path.join(dir_, str(int(num) + 1) + '_' + str(h))
        print('saving to: {}'.format(dir_))
        self.final_dir = dir_
        os.makedirs(self.final_dir, exist_ok=True)

    def training_callbacks(self, using_val=True):

        formatted_name = 'weights.{epoch:02d}-{val_loss:.4f}.hdf5'

        filepath = os.path.join(self.final_dir, formatted_name)
        if using_val:
            return [ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False,
                                    save_weights_only=True),
                    EarlyStopping(patience=25, restore_best_weights=True)]
        else:
            raise NotImplementedError('using_val=={}'.format(using_val))

    def save(self, file_name=None):
        if file_name is None:
            file_name = datetime.now().strftime('%Y%m%d_%H%M')

        out = os.path.join(self.final_dir, file_name)
        print('saving: {}'.format(out))
        self.model.save_weights(out)

    def load(self, file_name):
        self.model.load_weights(os.path.join(self.final_dir, file_name))

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)
