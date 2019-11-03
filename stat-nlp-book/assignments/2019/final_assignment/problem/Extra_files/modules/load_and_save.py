import re
import numpy as np
import os
from os.path import join
from os import listdir
import re
from collections import defaultdict

def save_to_ann(data, datadfir):
    """
    This function saves the relations (and only these) to a file.
    Args:
        data: {'txt': {'relations': [('R5', 'Hyponym', 'T14', 'T17'), ('*', 'Synonym', 'T20', 'T19')]}}
        datadir: The directory to save to, e.g. data/scienceie/predictions
    """

    for txt, df in data.items():

        #print(df.keys())

        ### Extract raw lines from .ann but without hyponyms and synonyms
        ann_content = df['raw_ann_load']
        lines_wo_rel = [line for line in ann_content if line[0] == 'T']

        ### Extract relations
        relations = df['relations']
        hyponyms = [rel for rel in relations if rel[1] == 'Hyponym']
        synonyms = [rel for rel in relations if rel[1] == 'Synonym']

        ### Create the correct lines
        hyp_lines = [hyp[0] + '\tHyponym-of Arg1:' + str(hyp[2]) + ' Arg2:' + str(hyp[3]) + '\t\n' for hyp in hyponyms]
        syn_lines = ['*\tSynonym-of ' + str(syn[2]) + ' ' + str(syn[3]) + '\n' for syn in synonyms]
        all_rel_lines = hyp_lines + syn_lines

        ### Create folder if doesn't exist
        prediction_dir = datadfir
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)

        ### Save the lines
        all_lines = lines_wo_rel + all_rel_lines
        fname = join(datadfir, txt + '.ann')
        with open(fname, 'w') as file:
            for line in all_lines:
                file.write(line)

    return

def load_scienceie(datadir):
    """
    Load the ScienceIE dataset from a given directory and return it in IOB format.
    Args:
        datadir: The directory to read from, e.g. data/scienceie/train or data/scienceie/dev
    Returns:
        A dictonary with the example format
        
        data['file_name'] = {'data': [tokens,IOBtags], #where tokens and IOBtags are lists with same length
                             'locations':locations, #A list of tuples with start-position and end-position of every token
                             'annotation_names': ann_names #A list of names ['T2','T1',...]
                             }

        remove_refs: Whether it should remove references (brackets with numbers)
    """

    txt_files  = [f for f in listdir(datadir) if f[-3:] == 'txt']
    
    
    try:
        ann_files = [f[:-3] + 'ann' for f in txt_files]
        ann_file_exists = True
    except:
        print('Cannot find annotation files. The returned data set cannot be used for training.')
        ann_file_exists = False
    
    #ann_file_exists = False
    
    data = {}
    n_rels_missed = 0
    total_rels = 0

    total_ann_lines = 0
    ann_lines_after_sort = 0
    ann_lines_after_id = 0

    for i in range(len(txt_files)):
        org_text = load_txt_str(txt_files[i], datadir)
        tokens, locations = split_txt_str(org_text)
        starting_locations = np.array([loc[0] for loc in locations])
        ending_locations = np.array([loc[1] for loc in locations])
        
        
        ### This creates the IOBtags (labels) for the training
        IOBtags = ['O']*len(tokens)
        
        relations = []
        my_ann_names = []
        
        if ann_file_exists:

            ann_content = load_ann_file(ann_files[i],datadir)
            total_ann_lines += len(ann_content)

            sorted_annotation = split_and_sort_ann(ann_content)
            ann_lines_after_sort += len(sorted_annotation)

            entity_types, entity_words, ann_names, entity_locations = ann_to_entities(sorted_annotation)
            
            for line in range(len(ann_names)):
                starting_loc = entity_locations[line][0]
                ending_loc = entity_locations[line][1]
                
                starting_idx_in_tokens = int(np.searchsorted(starting_locations, starting_loc, side = 'left')) #if equal, puts in front
                ending_idx_in_tokens = int(np.searchsorted(ending_locations, ending_loc, side = 'left')) #if equal, puts in front
                
                tokens_in_text = tokens[starting_idx_in_tokens:(ending_idx_in_tokens+1)]
                
                if tokens_in_text == entity_words[line]:
                    
                    ##If we can find it, add the T
                    my_ann_names.append(ann_names[line])
                    
                    #Insert the IOB-tags
                    IOBtags[starting_idx_in_tokens] = 'B-' + entity_types[line]
                    for idx in range(1,len(tokens_in_text)):
                        IOBtags[starting_idx_in_tokens + idx] = 'I-' + entity_types[line]


            #extract synonym/hyponyms and add them to data
            non_entity = [line for line in ann_content if line[0] != 'T']
            non_entity_split = [re.split(r'[\- :\t]', line.rstrip()) for line in non_entity]

            remove = ['','of','Arg1','Arg2']
            relations = [tuple(token for token in line if token not in remove) for line in non_entity_split]

            ### Only relations where the entities are in fact found (survived the overlap cleaning)
            rels_found = [rel for rel in relations if \
                          rel[2] in my_ann_names and \
                          rel[3] in my_ann_names]
                
        
        ### Check if we found all relations
        n_rels_missed += len(relations) - len(rels_found)
        total_rels += len(relations)

        #if len(relations) - len(rels_found) > 0:
        #    print("Missing relations in: {}".format(txt_files[i]))

        ann_lines_after_id += len(my_ann_names)

        #Add to output dict
        data[txt_files[i][:-4]] = {'tokens': tokens,
                                   'IOBtags': IOBtags,
                                   'locations':locations,
                                   'annotation_names': my_ann_names,
                                   'relations': rels_found,
                                   'raw_ann_load' : ann_content}

    print("Number of entities removed due to overlap: {} out of {}".format(total_ann_lines - ann_lines_after_sort, total_ann_lines))
    print("Number of entities not identified in text: {} out of {}".format(ann_lines_after_sort - ann_lines_after_id, ann_lines_after_sort))
    print("Number of relations lost due to overlap: {} out of {}".format(n_rels_missed, total_rels))

    #Remov references and concatenate
    data_wo_refs = _remove_refs(data)
    print("Removed references.")
    data_concat = _concatenate_ie_eg(data_wo_refs)
    print("Concatenated 'i.e.' and 'e.g.'.")

    return data_concat


def reformat_to_save(data_w_metadata):
    '''
    This transforms the data back from tf-output to dev_data

    Input: data_w_metadata = {'metadata':[],data_Y:[]}
    Output: outdata = {'txt': {'relations': [('R5', 'Hyponym', 'T14', 'T17'), ('*', 'Synonym', 'T20', 'T19')]}}
    '''

    #### 

    # Subset only those which are not NONE
    data_Y = np.array(data_w_metadata['data_Y'])
    mask = (data_Y != 'NONE')

    metadata = np.array(data_w_metadata['metadata'])[mask]
    data_Y = np.array(data_w_metadata['data_Y'])[mask]

    # Reshape to the dictionary: 
    # data_dict[txt] = [[T1,T2,'Synonym'],[T3,T4,'Hyponym']]
    tmp = [(metadata[idx][0], [metadata[idx][1],metadata[idx][2],data_Y[idx]]) for idx in range(len(data_Y))]
    data_dict = defaultdict(list)
    for k, v in tmp:
        data_dict[k].append(v)


    #### Add the relations file for file
    # Initialize outdata (would normaly create defaultdict but need all txt's)
    all_txt_files = np.unique(np.array([meta[0] for meta in data_w_metadata['metadata']]))
    outdata = {txt: [] for txt in all_txt_files}
    
    for txt, relations in data_dict.items():
        
        ####  Add the tags in fron (*,R1,R2 osv.)
        # Convert hyponym_reverted to hyponym
        rel_hyponyms = [[x[0], x[1]] for x in relations if x[2] == 'Hyponym']
        rel_hyponyms += [[x[1], x[0]] for x in relations if x[2] == 'Hyponym_reverted']

        # Add R1, R2 osv.
        rel_hyponyms = [('R' + str(idx+1), 'Hyponym', rel_hyponyms[idx][0], rel_hyponyms[idx][1]) for idx in range(len(rel_hyponyms))]
        # Add *
        rel_synonyms = [('*', 'Synonym', x[0], x[1])for x in relations if x[2] == 'Synonym']

        all_relations = rel_hyponyms + rel_synonyms

        outdata[txt] = all_relations

    return outdata

def _remove_refs(data):
    counter = 0
    for txt in data:
        df = data[txt]

        tokens = df['tokens']
        IOBtags = df['IOBtags']

        refs = []
        remove_idxs = []

        start_brack_idxs = [i for i, x in enumerate(tokens) if x == "["]
        end_brack_idxs = [i for i, x in enumerate(tokens) if x == "]"]

        #Check equal amount of '[' and ']' (otherwise it would break)
        if len(start_brack_idxs) != len(end_brack_idxs):
                continue

        #Check if brackets are references
        for i in range(len(start_brack_idxs)):
            start_brack_idx = start_brack_idxs[i]
            end_brack_idx = end_brack_idxs[i]

            #Check if brackets is purely references
            brack_is_ref = True
            for i in range(start_brack_idx+1,end_brack_idx):
                if tokens[i] not in [',','–','-'] and not tokens[i].isdigit():
                    brack_is_ref = False
                    break

            #If brack is ref
            if brack_is_ref:
                remove_idxs.append([i for i in range(start_brack_idx, end_brack_idx+1)])
                refs.append([tokens[i] for i in range(start_brack_idx, end_brack_idx+1)])

        #Ensure that they are not entities (otherwise we naturally should not remove it)
        for ref_number, ref_idxs in enumerate(remove_idxs):

            for idx in ref_idxs:
                if IOBtags[idx] != 'O':
                    '''
                    print("Will not remove this:")
                    print(txt)
                    print(refs[ref_number])
                    print("---------------------")
                    '''
                    del remove_idxs[ref_number]
                    break

        #Remove the references
        flat_list_idxs = [idx for ref in remove_idxs for idx in ref]
        remove_reversed_idxs = np.sort(np.array(flat_list_idxs,dtype = int))[::-1]
        for idx in remove_reversed_idxs:
            del df['tokens'][idx]
            del df['IOBtags'][idx]
            del df['locations'][idx]

        #Hvis der rent faktisk bliver slettet, print så lige text, så jeg kan sammenligne
        if len(remove_reversed_idxs) > 0:
            counter +=1
            #print(txt)
        
        #To be ceartain everything is fine
        assert len(df['tokens']) == len(df['IOBtags'])
        assert len(df['IOBtags']) == len(df['locations'])
        
        Bs = [x for x in df['IOBtags'] if x[0] == 'B']
        assert len(Bs) == len(df['annotation_names'])
        
    return data

def load_txt_str(filename, datadir):
    with open(join(datadir,filename), 'r') as f: #open the file
        contents = f.readlines()
        assert len(contents) == 1
    return contents[0]

def split_txt_str(text_string):
        newline_striped = text_string.rstrip() #Remove newline
        split_str = re.split('(\W)', newline_striped) #Split on everything but words
            
        #Remove empty string and function application:
        for i in range(len(split_str)-1,-1,-1):
            if split_str[i] in ['','\u2061']:
                del split_str[i]

        #Save the location, so it can be put back.   
        assert(len(split_str) >=1)
        
        str_lengths = np.array([len(token) for token in split_str])
        #assert(len(str_lengths) >= 1)

        begin_char = np.append(np.array([0]),np.cumsum(str_lengths[:-1]))
        end_char = np.cumsum(str_lengths)
        locations = [(begin_char[i],end_char[i]) for i in range(len(begin_char))]
 
        for i in range(len(split_str)-1,-1,-1):
        #Remove space, non-breaking space, zero-width-space, thin space:
            if split_str[i] in [' ','\xa0','\u200b','\u2009']:
                del split_str[i]
                del locations[i]

        return split_str, locations

def load_ann_file(filename, datadir):
    #open the file
    with open(join(datadir,filename), 'r',newline = '\n') as f:
        contents = f.readlines()
    return contents

def split_and_sort_ann(contents):
    
    #Only keep text-bound annotation
    only_entity = [line for line in contents if line[0] == 'T'] 

    splittet = [line.rstrip().split('\t') for line in only_entity]
    for line in splittet:
        line[1] = line[1].split(' ')
        line[1][1] = int(line[1][1])
        line[1][2] = int(line[1][2])
            
    ### Order according to start of label, with highest end comming first
    sort_indices = np.argsort(np.array([line[1][1] + 1/(2+line[1][2]) for line in splittet]))
    sorted_annotation = [splittet[i] for i in sort_indices]
        
    ### Remove double occurences or overlapping labels from annotation files
    label_number = 1
    while True:
        if label_number >= len(sorted_annotation):
            break

        #If next label starts before the last ends
        if sorted_annotation[label_number-1][1][2] > sorted_annotation[label_number][1][1]:
            del sorted_annotation[label_number]
        else:
            label_number +=1

    
    return sorted_annotation

def ann_to_entities(annotations):
    
    for line in annotations:
        line[2],_ = split_txt_str(line[2])
    
    entity_types = [line[1][0] for line in annotations]
    entity_words = [line[2] for line in annotations]
    ann_names = [line[0] for line in annotations]
    entity_locations = [(int(line[1][1]),int(line[1][2])) for line in annotations]
    
    return entity_types, entity_words, ann_names, entity_locations


def _concatenate_ie_eg(data):
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