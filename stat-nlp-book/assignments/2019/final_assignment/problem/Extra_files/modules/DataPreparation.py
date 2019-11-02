# DATA PREPARATION
from statnlpbook.vocab import Vocab
from gensim.models import fasttext
from gensim.models import KeyedVectors
import copy
import numpy as np

def entityLocator(indata):
    # line for each T: T, start, end, entity
    entities = []
    name_counter = 0
    i = 0
    flag = 0
    for tag in indata['IOBtags']:
        if tag[0] == 'B':
            name = indata['annotation_names'][name_counter]
            tg = tag
            start = i
            name_counter = name_counter + 1
            flag = 1 # flag if we are currently inside iob
        if tag[0] == 'O' and flag == 1:
            # if entity ended - submit to entities
            end = i-1
            entities.append((name, tg, start, end))
            flag = 0
        #end-if
        i = i + 1
    #end-for
    if flag == 1:
        #submit
        entities.append((name, tg, start, i-1))
        flag = 0
    #end-tricks
    return entities
#end-def

def getPOS(data_dic, nlp):
    return [[token.pos_ for token in sent] for sent in nlp.pipe([data_dic['tokens']])][0]
#end-def

def addPOStoDic(data_dic, nlp):
    pos = getPOS(data_dic, nlp) # do not override default!
    # assert lenght are equal - otherwise shit has hit the fan!
    assert len(pos) == len(data_dic['tokens']), 'POS length does not match token lengths!'
    data_dic['POS'] = pos
    pass

def getLemma(data_dic, nlp):
    return [[token.lemma_ for token in sent] for sent in nlp.pipe([data_dic['tokens']])][0]
#end-def

def addLemmatoDic(data_dic, nlp):
    lemma = getLemma(data_dic, nlp) # do not override default!
    # assert lenght are equal - otherwise shit has hit the fan!
    assert len(lemma) == len(data_dic['tokens']), 'Lemma length does not match token lengths!'
    data_dic['Lemma'] = lemma
    pass

# we create input pair given entities from entitylocator and data

def inputPair(entityA, entityB, indata, nlp):
    out = []
    # get distances
    _, _, A_start, A_end = entityA
    _, _, B_start, B_end = entityB
    
    # check if POS
    if not 'POS' in indata.keys():
        # warn user
        print('No POS in data dictionary - adding them...')
        # add them
        addPOStoDic(indata, nlp)
        print('POS added to data dictionary.')

    # check if LEMMA
    if not 'Lemma' in indata.keys():
        # warn user
        print('No Lemma in data dictionary - adding them...')
        # add them
        addLemmatoDic(indata, nlp)
        print('Lemma added to data dictionary.')
    
    # Get tokens inbetween
    i = 0
    for token in indata['tokens']:
        # A < B
        if i in range(A_start, B_end+1):
            relA = max(i-A_end, 0)
            relB = min(i-B_start,0)
            # get POS
            out.append([token, relA, relB, indata['IOBtags'][i], indata['POS'][i], indata['Lemma'][i] ])
        # B < A
        if i in range(B_start, A_end+1):
            relA = max(A_start-i, 0)
            relB = min(B_end-i,0)
            # get POS
            out.append([token, relA, relB, indata['IOBtags'][i], indata['POS'][i], indata['Lemma'][i] ])
        #end-if
        i = i+1
    #end-for
    return out
#end-def

# SETUP PRE-TRAINED-WORD-EMBEDDINGS

# Create model
class WordEmbedder:
    def __init__(self):
        self.vmodel = None
        self.length = None

    def buildModel(self, limit = 3000, pathToCreateModel = "engmodel.model", pathToLargeModel = 'wiki-news-300d-1M.vec'):
        engmodel = KeyedVectors.load_word2vec_format(pathToLargeModel, limit=limit)
        engmodel.save(pathToCreateModel)
        self.vmodel = engmodel
        self.length = len(self.vmodel.vectors[0])

    def loadModel(self, pathToModel = "engmodel.model"):
        self.vmodel = KeyedVectors.load(pathToModel)
        self.length = len(self.vmodel.vectors[0])
        
    def getEmbedding(self, word):
        if word in self.vmodel.vocab:
            return self.vmodel[word]
        else:
            return np.zeros(self.length)
#end-class

def GenerateVocabs(fulldata, maxdistance, maxsize, nlp):
    maxlen = maxdistance

    fulldata = copy.deepcopy(fulldata)
    # Create POS for all before we create vocab
    for entry in list(fulldata.values()):
        addPOStoDic(entry, nlp)
        addLemmatoDic(entry, nlp)
    # we introduce four vocabs: word, dist, entity, pos
    vocab_w = Vocab.from_iterable([fulldata[i]['tokens'] for i in fulldata], max_size = maxsize)
    vocab_dist = Vocab.from_iterable(range(-maxlen, maxlen))
    vocab_ent = Vocab.from_iterable([fulldata[i]['IOBtags'] for i in fulldata]) # these should have all
    vocab_pos = Vocab.from_iterable([fulldata[i]['POS'] for i in fulldata]) # and so should these!
    vocab_lemma = Vocab.from_iterable([fulldata[i]['Lemma'] for i in fulldata]) # and so should these!
    return vocab_w, vocab_dist, vocab_ent, vocab_pos, vocab_lemma
#end-def

# Convert input from strings to ints for embedding layer (or use pre-trained embeddings):
def createX(xdata, vocab_words, vocab_distances, vocab_entities, vocab_pos, vocab_lemma):
    out = [[vocab_words.map_to_index([w[0]])[0],                # token
            vocab_distances.map_to_index([w[1]])[0],            # dist 1
            vocab_distances.map_to_index([w[2]])[0],            # dist 2
             vocab_entities.map_to_index([w[3]])[0],            # entities
             vocab_pos.map_to_index([w[4]])[0],                 # pos
             vocab_lemma.map_to_index([w[5]])[0]] for w in xdata] #lemma
    return out
#end-def

def createXEmbeddings(xdata, embedder, vocab_distances, vocab_entities, vocab_pos):
    out = [[embedder.getEmbedding(w[0]),
            vocab_distances.map_to_index([w[1]])[0],
            vocab_distances.map_to_index([w[2]])[0],
             vocab_entities.map_to_index([w[3]])[0],
             vocab_pos.map_to_index([w[4]])[0],
             embedder.getEmbedding(w[5])] for w in xdata]
    return out
#end-def