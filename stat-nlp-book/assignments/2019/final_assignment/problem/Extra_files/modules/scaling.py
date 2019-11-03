import numpy as np
from copy import deepcopy


def upscale(data):
    pass

def downscale(data, *vocab_w):
    '''
    This can potentially take several downclae precedures and apply iteratively
    '''

    # Remove dots (inplace removal)
    data = _remove_dot_in_sentence(data, vocab_w)

    return data


def _remove_dot_in_sentence(data_m_XY, *vocab_w):

    ## Check if indices == True
    '''
    data_X = data_m_XY['data_X']
    words = []
    for data_point in data_X:
        words += [x[0] for x in data_point]
    all_ints = all([isinstance(x,int) for x in words])
    '''
    
    ## Extract data_Y and metadata
    data_Y = data_m_XY['data_Y']
    metadata = data_m_XY['metadata']

    ## Extract data_X
    if len(vocab_w[0]) > 1:
        raise Exception("Too many arguments")
    elif len(vocab_w[0]) == 0:
        data_X = data_m_XY['data_X']
    else:
        vocab_w = vocab_w[0][0]
        #Create new 'data_X
        data_X = []
        for data_point in data_m_XY['data_X']:
            new_X_words = []
            for x in data_point:
                new_X_words.append([vocab_w.get_label(x[0])] + list(x[1:]))
            data_X.append(np.array(new_X_words))
        
    
    ## Initialize for the loop
    new_metadata = []
    new_data_Y = []
    new_data_X = []

    ## Loop
    for idx in range(len(data_Y)):    

        # Tjek if '.' in words, that next words is in range, and this starts with uppercase
        words = [x[0] for x in data_X[idx]]
        if '.' in words:
            next_word_idx = words.index('.') + 1
            if next_word_idx < len(words):
                next_word = words[next_word_idx]
                if next_word[0].isupper():
                    continue
        
        #Append if not 'continue'
        new_metadata.append(metadata[idx])
        new_data_Y.append(data_Y[idx])
        new_data_X.append(data_X[idx])

    ## Replace back
    data_m_XY['metadata'] = new_metadata
    data_m_XY['data_Y'] = new_data_Y
    data_m_XY['data_X'] = new_data_X

    print("Kept {} datapoints out of {} after downscaling.".format(len(new_metadata), len(data_Y)))

    return data_m_XY
