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

    def buildModel(self, limit = 3000, pathToCreateModel = "engmodel.model", pathToFastText = 'wiki-news-300d-1M.vec'):
        engmodel = KeyedVectors.load_word2vec_format(pathToFastText, limit=limit)
        # save and set model
        engmodel.save(pathToCreateModel)
        self.vmodel = engmodel
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

    def fitPCA_viaVocab(self, mastervocab, n_components):
        '''
        This functions extracts all words form vocab and fits the PCA with those
        '''
        # get all words from vocab:
        words = [mastervocab.word_vocab.get_label(i) for i in range(len(mastervocab.word_vocab))]
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