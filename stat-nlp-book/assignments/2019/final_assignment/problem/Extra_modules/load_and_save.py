import re
import numpy as np
import os
from os.path import join
from os import listdir
import re

def save_to_ann(data, datadfir):
    """
    Save annotations in IOB format back to .ann files.
    Args:
        data: The annotations in IOB format
        datadir: The directory to save to, e.g. data/scienceie/predictions
    """
    file_names = list(data.keys())
    
    for file_name in file_names:
        file = data[file_name]
        
        #If two 'B-...' arrives, begin then new, but if 'I-...' after 'O', let it be 'B-...'
        
        tokens = np.array(file['tokens'])
        IOBtags = np.array(file['IOBtags'])
        locations = np.array(file['locations'])

        #Put into one
        ann_tags = []
        ann_names = file['annotation_names'] #Maybe None
        ann_locations = []
        ann_entities = []

        #IOBtag (str), locations (tuple), entity (list with start stop at every)
        
        tmp = []
        last_tag = 'O'
        
        for index in range(len(tokens)):
            
            tag = IOBtags[index]
            word = tokens[index]
            location = tuple(locations[index])
            
            if tag == 'O':
                pass
            elif tag[0] == 'B':
                ann_tags.append(tag[2:])
                tmp.append([[word,location]])
            elif tag[0] == 'I' and last_tag == 'O': #Begin new_line
                ann_tags.append(tag[2:])
                tmp.append([[word,location]])
            elif tag[0] == 'I' and last_tag[2:] != tag[2:]: #Begin new_line
                ann_tags.append(tag[2:])
                tmp.append([[word,location]])
            elif tag[0] == 'I' and last_tag[2:] == tag[2:]: #Continue this line
                tmp[-1].append([word,location])
            else:
                print('Tag: ',tag)
                print('Last_tag: ',last_tag)
                raise Exception("This should not happen")
                
            last_tag = tag


        sentences = []
        for sentence in tmp:
            ann_locations.append((sentence[0][1][0],sentence[-1][1][1]))

            entity_str = ''

            n=0

            while True:
                entity_str += sentence[n][0]
                n +=1

                if len(sentence) <= n:
                    break

                n_spaces = sentence[n][1][0] - sentence[n-1][1][1]
                
                try:
                    assert (n_spaces in [0,1])
                except:
                    n_spaces = 1
                    
                for i in range(n_spaces):
                    entity_str += ' '

            sentences.append(entity_str)

        if ann_names is None:
            ann_names = ['T' + str(i) for i in range(1,len(sentenses)+1)]
        
        ann_sorted = [[ann_names[i],
                       [ann_tags[i],ann_locations[i][0],ann_locations[i][1]],
                       sentences[i]] for i in range(len(ann_names))]
        
        #Re-sort it back. (Only makes difference if we have ann_names)
        sort_indices = np.argsort(np.array([int(line[0][1:]) for line in ann_sorted]))
        org_sorting = [ann_sorted[i] for i in sort_indices]
        
        #Concatenate 
        for line in org_sorting:
            line[1] = line[1][0] + ' ' + str(line[1][1]) + ' ' + str(line[1][2])

        #Save the bloody thing
        prediction_dir = datadfir
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)

        fname = join(datadfir,file_name + '.ann')
        with open(fname,'w') as file:
            for line in org_sorting:
                tmp = '\t'.join(line)
                tmp += '\n'
                file.write(tmp)

    return


def load_scienceie(datadir,remove_refs = True):
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
    counter = 0

    total_ann_lines = 0
    ann_lines_after_sort = 0
    ann_lines_after_id = 0

    for i in range(len(txt_files)):
        org_text = load_txt_str(txt_files[i],datadir)
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
            non_entity_split = [re.split(r'[\- :\t]',line.rstrip()) for line in non_entity]

            remove = ['','of','Arg1','Arg2']
            relations = [tuple(token for token in line if token not in remove) for line in non_entity_split]

            ### Only relations where the entities are in fact found (survived the overlap cleaning)
            rels_found = [rel for rel in relations if \
                          rel[2] in my_ann_names and \
                          rel[3] in my_ann_names]
                
        
        ### Check if we found all relations
        if len(rels_found) != len(relations):
            counter += 1
        ann_lines_after_id += len(my_ann_names)
        #Add to output dict
        data[txt_files[i][:-4]] = {'tokens': tokens,
                                   'IOBtags': IOBtags,
                                   'locations':locations,
                                   'annotation_names': my_ann_names,
                                   'relations': relations}

    print("Number of lines removed due to overlap: {} out of {}".format(total_ann_lines - ann_lines_after_sort, total_ann_lines))
    print("Number of lines not identified in text: {} out of {}".format(ann_lines_after_sort - ann_lines_after_id, ann_lines_after_sort))

    if remove_refs:
        data_wo_refs = _remove_refs(data)
        print("The function removed references from data. If not desired, choose 'remove_refs = False'.")
        return data_wo_refs
    else:
        return data

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
                if tokens[i] != ',' and not tokens[i].isdigit():
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
