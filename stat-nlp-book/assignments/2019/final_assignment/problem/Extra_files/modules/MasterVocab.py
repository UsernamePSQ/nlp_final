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


        # TESTING SHIT
        def Testfunction55(dev_data, nlp):
            # Quick-test cell!
            if False:
                tst_vocab = Vocab.from_iterable(["B-Process"], max_size = 3)
                print(tst_vocab.map_to_index(["B-Process"]))
                A = tst_vocab.map_to_index(["B-Process"])
                print(tst_vocab.get_label(A[0]))
                
            if False:
                vocab = MasterVocab(max_vocab_size = 4092)
                vocab.generateVocabularies(dev_data, nlp)
                dat = dev_data[list(dev_data.keys())[0]]
                entities = entityLocator(dat)
                xdata = inputPair(entities[0], entities[1], dat, nlp)
                print(xdata)
                x2data = vocab.transformX_toIndex(xdata)
                print(x2data)
                print(vocab.transformX_toLabel(x2data))
        pass