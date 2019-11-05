# DATA PREPARATION
from statnlpbook.vocab import Vocab
from Extra_files.modules.WordEmbedder import WordEmbedder
import copy
import numpy as np

def entityLocator(indata):
    # line for each T: T, start, end, entity
    entities = []
    name_counter = 0
    i = 0
    flag = 0
    for tag in indata['IOBtags']:
        # fix for B without O inbetween
        if tag[0] == 'B':
            if flag == 0:
                name = indata['annotation_names'][name_counter]
                tg = tag
                start = i
                name_counter = name_counter + 1
                flag = 1 # flag if we are currently inside iob
            else:
                # This is the case where a B is right next to another B or I of another sort
                # append before updating
                end = i-1
                entities.append((name, tg, start, end))
                flag = 0
                # setup new one
                name = indata['annotation_names'][name_counter]
                tg = tag
                start = i
                name_counter = name_counter + 1
                flag = 1 # flag if we are currently inside iob
            #end-if
        #end-if

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