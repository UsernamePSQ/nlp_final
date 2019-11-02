

def concatenate_ie_eg(data):
    '''
    This function makes 'i.e.' and 'e.g.' to a single word.
    '''

    for txt in data:

        #Extract the relevant
        data_point = data[txt]
        tokens = data_point['tokens']
        IOBtags = data_point['IOBtags']
        ann_names = data_point['annotation_names']
        locations = data_point['locations']

        ### Loop
        new_tokens = []
        new_IOBtags = []
        new_locations = []

        token_idx = 0

        while token_idx <= len(tokens)-4:
            new_IOBtags.append(IOBtags[token_idx])
            new_locations.append(locations[token_idx])

            # Don't do anything if it is part of a tag
            IOBtag_valid = True
            if IOBtags[token_idx:(token_idx+4)] != ['O']*4:      
                IOBtag_valid = False

            if tokens[token_idx:(token_idx+4)] == ['e', '.', 'g', '.']:
                eg = True
                ie = False
            elif tokens[token_idx:(token_idx+4)] == ['i', '.', 'e', '.']:
                eg = False
                ie = True
            else:
                eg = False
                ie = False

            if IOBtag_valid and (eg or ie):
                conc_abr = ''.join(tokens[token_idx:(token_idx+4)])
                new_tokens.append(conc_abr)
                token_idx += 4
            else:
                new_tokens.append(tokens[token_idx])
                token_idx += 1
                
            # End if
        # End while
        new_tokens += tokens[token_idx:len(tokens)]
        new_locations += locations[token_idx:len(tokens)]
        new_IOBtags += IOBtags[token_idx:len(tokens)]

        assert len(new_tokens) == len(new_locations)
        assert len(new_locations) == len(new_IOBtags)
        '''
        if len(new_tokens) < len(tokens):
            print(txt)
        '''
        data_point['tokens'] = new_tokens
        data_point['locations'] = new_locations
        data_point['IOBtags'] = new_IOBtags

    #End txt loop

    return data
