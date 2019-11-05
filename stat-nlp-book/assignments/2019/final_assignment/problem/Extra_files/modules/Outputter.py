# Constructs and checks if current setup of relations is legal

import numpy as np
from copy import deepcopy

def checkLegality(relations):
    # this one should return whether or not the current relations are legal
    
    # example functionality so far
    labels = [np.argmax(rel) for rel in relations]
    
    # check if two of the same
    counts = np.unique(labels, return_counts = True)[1]
    if np.max(counts) > 1:
        return False
    #end-if
    return True
#end-def

def mostProbableYetLegal(predictions):
    # relations should be an array of probabilities
    legalRelations = []
    checkRelations = []
    # ensure that relations are np.array
    #predictions = np.array(predictions)
    # choose relations one-by-one ranked by highest prob
    probs = np.max(predictions, axis = 1)
    highestorder = np.argsort(probs)[::-1] #flip the order
    # sorted
    pred_sorted = predictions[highestorder]
    for pred in pred_sorted:
        checkRelations = deepcopy(legalRelations)
        checkRelations.append(pred)
        # append if legal
        if checkLegality(checkRelations):
            legalRelations.append(pred)
        #end-if
    #end-for    
    return legalRelations
#end-def

