from Extra_files.modules.DataPreparation import addPOStoDic, addLemmatoDic, entityLocator, inputPair
import copy
from statnlpbook.vocab import Vocab
import warnings

class MasterVocab:
    def __init__(self, max_vocab_size):
        self.max_size = max_vocab_size
        self.word_vocab = None
        self.pos_vocab = None
        self.entities_vocab = None
    def generateVocabularies(self, data, nlp_language_class):
        # call function
        data = copy.deepcopy(data)
        # Create POS for all before we create vocab
        for entry in list(data.values()):
            addPOStoDic(entry, nlp_language_class)
            addLemmatoDic(entry, nlp_language_class)
        # we introduce four vocabs: word, dist, entity, pos
        self.word_vocab = Vocab.from_iterable([data[i]['tokens'] for i in data], max_size = self.max_size)
        self.entities_vocab = Vocab.from_iterable([data[i]['IOBtags'] for i in data], max_size = 50) 
        self.pos_vocab = Vocab.from_iterable([data[i]['POS'] for i in data], max_size = 50)
        # Voila!
    def transformX_toIndex(self, Xdata):
        if not self.word_vocab is None:
            out = [[self.word_vocab.map_to_index([w[0]])[0],                # token
                    w[1],                                                   # dist 1
                    w[2],                                                   # dist 2
                    self.entities_vocab.map_to_index([w[3]])[0],            # entities
                    self.pos_vocab.map_to_index([w[4]])[0],                 # pos
                    self.word_vocab.map_to_index([w[5]])[0]] for w in Xdata] #lemma
        else:
            warnings.warn("WARNING: MasterVocab does not have a generated vocab! Returning original input...")
            out = Xdata
        return out

    def transformX_toEmbeddings(self, Xdata, embedder):
        if not self.word_vocab is None:
            out = [[embedder.getEmbedding(w[0]), # w token
                    w[1], # dist 1
                    w[2], # dist 2
                    self.entities_vocab.map_to_index([w[3]])[0], #ent
                    self.pos_vocab.map_to_index([w[4]])[0], #pos
                    embedder.getEmbedding(w[5])] for w in Xdata] # lemma
        else:
            warnings.warn("WARNING: MasterVocab does not have a generated vocab! Returning original input...")
            out = Xdata
        return out
    #end-def

    def transformX_toLabel(self, Xdata):
        if not self.word_vocab is None:
            out = [[self.word_vocab.get_label(w[0]),                 # token
                    w[1],                                            # dist 1
                    w[2],                                            # dist 2
                    self.entities_vocab.get_label(w[3]),             # entities
                    self.pos_vocab.get_label(w[4]),                  # pos
                    self.word_vocab.get_label(w[5])] for w in Xdata] #lemma
        else:
            warnings.warn("WARNING: MasterVocab does not have a generated vocab! Returning original input...")
            out = Xdata
        return out
    #end-def
#end-class