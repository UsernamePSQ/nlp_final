from gensim.models import fasttext
from gensim.models import KeyedVectors
import numpy as np
from sklearn.decomposition import PCA
from pickle import dump, load

class WordEmbedder:
    def __init__(self):
        self.vmodel = None
        self.length = None
        self.pcaModel = None
        self.pcaLength = None

    def buildModel(self, limit = 15000, pathToCreateModel = "Extra_files/resources/engmodel.model", pathToFastText = 'wiki-news-300d-1M.vec'):
        engmodel = KeyedVectors.load_word2vec_format(pathToFastText, limit=limit)
        # save and set model
        engmodel.save(pathToCreateModel)
        self.vmodel = engmodel
        self.length = len(self.vmodel.vectors[0])

    def buildModel_viaVocab(self, masterVocab, limit_ours = 15000, limit_large = 150000, info = True,
                   pathToCreateModel = "Extra_files/resources/engmodel.model", pathToFastText = 'wiki-news-300d-1M.vec'):
        # extract words from vocab
        words = [masterVocab.word_vocab.get_label(i) for i in range(len(masterVocab.word_vocab))]
        # load in the two models
        ftmodel = KeyedVectors.load_word2vec_format(pathToFastText, limit=limit_large) # massive guy
        ourmodel = KeyedVectors.load_word2vec_format(pathToFastText, limit=limit_ours) # our guy
        # go through each word in vocab and add if possible
        counter = 0
        for w in words:
            if w not in ourmodel.vocab and w in ftmodel.vocab:
                counter = counter + 1
                # add it
                ourmodel.add(w, ftmodel[w])
        #end-for
        print("Model built. Added a total of counter %s words to our vocab from larger model" % (counter))
        if info == True:
            counter = 0
            for w in words:
                if w in ourmodel.vocab:
                    counter = counter + 1
                #end-if
            #end-for
            print("MasterVocab size: %s, MasterVocab words in our FastText Vocab %s, coverage: %s" % (len(words), counter, (counter+0.0)/(0.0+len(words))))
        pass

        # save and set model
        ourmodel.save(pathToCreateModel)
        self.vmodel = ourmodel
        self.length = len(self.vmodel.vectors[0])

    def loadModel(self, pathToModel = "Extra_files/resources/engmodel.model"):
        self.vmodel = KeyedVectors.load(pathToModel)
        self.length = len(self.vmodel.vectors[0])

    def fitPCA(self, words, n_components):
        '''
        This function fits a PCA model from input words and number of PCA components.
        '''
        # setup pca
        pca = PCA(n_components = n_components)
        # get word embeddings for words
        embeddings = [self.getEmbedding_prePCA(word) for word in words]
        # fit it
        pca.fit(embeddings)
        self.pcaModel = pca
        self.pcaLength = n_components

    def fitPCA_viaVocab(self, mastervocab, n_components, include_oov = False):
        '''
        This functions extracts all words form vocab and fits the PCA with those
        '''
        # get all words from vocab:
        words = [mastervocab.word_vocab.get_label(i) for i in range(len(mastervocab.word_vocab))
                 if mastervocab.word_vocab.get_label(i) in self.vmodel.vocab]
        self.fitPCA(words, n_components)
        # et Voila!

    def savePCA(self, path = "Extra_files/resources/PCA.pkl"):
        dump(self.pcaModel, open(path,"wb"))
        print("PCA model saved at %s" % (path))
    
    def loadPCA(self, path = "Extra_files/resources/PCA.pkl"):
        self.pcaModel = load(open(path, 'rb'))
        self.pcaLength = self.pcaModel.n_components_ # get n_components
        print("PCA model loaded from %s" % (path))

    def loadEmbedder(self, 
                     pathModel = "Extra_files/resources/engmodel.model",
                     PathPCA = "Extra_files/resources/PCA.pkl"):
        self.loadModel(pathModel)
        self.loadPCA(PathPCA)

    def getEmbedding_prePCA(self, word, indicator = False):
        '''
        This function is used to get naive embeddings from FastText before PCA is fit
        '''
        assert self.vmodel is not None
        word = word.lower()
        if word in self.vmodel.vocab:
            out = self.vmodel[word]
            if indicator:
                out = np.append(out, 0.0)
            return out
        else:
            out = np.zeros(self.length)
            if indicator:
                out = np.append(out, 1.0)
            return out
        
    def getEmbedding(self, word):
        '''
        This function requires both fasttext model and PCA
        '''
        assert self.vmodel is not None
        assert self.pcaModel is not None
        word = word.lower()
        if word in self.vmodel.vocab:
            out = self.pcaModel.transform([self.vmodel[word]])[0] # expects two dimensional input
            out = np.append(out, 0.0)
            return out
        else:
            out = self.pcaModel.transform([np.zeros(self.length)])[0]
            out = np.append(out, 1.0)
            return out
#end-class