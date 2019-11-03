import numpy as np

def add_UMLS(data):

    # Load raw file
    with open("Extra_files/data/UMLS_relations.txt","r") as file:
        tmp = file.readlines()[0]
        
    # Restructure
    split = tmp.split('|')
    del split[-1]
    assert len(split) % 3 == 0
    relations = np.array(split).reshape((int(len(split) / 3),3))

    # Extract hyponyms ('isa')
    hyponyms = [(rel[0].lower(),rel[2].lower()) for rel in relations if rel[1] == 'isa']

    ### Extract entity_pairs
    entity_pairs = _extract_entity_pairs(data['data_X'])
    #print(hyponyms)

    data['UMLS'] = []
    for data_idx in range(len(data['data_X'])):

        ent_pair = entity_pairs[data_idx]

        if any([((hyp[0] in ent_pair[0]) and (hyp[1] in ent_pair[1])) for hyp in hyponyms]):
            data['UMLS'].append('Hyponym')
        elif any([((hyp[1] in ent_pair[0]) and (hyp[0] in ent_pair[1])) for hyp in hyponyms]):
            data['UMLS'].append('Hyponym_reverted')
        else:
            data['UMLS'].append('NONE')
    return data

    
### Bonus function for adding UMLS
def _extract_entity_pairs(data_X):
    entity_pairs = []
    for data_point in data_X:

        ### Extract sentence between entities ###
        length_first_entity = [x[1] for x in data_point].index('1')
        begin_last_entity = [x[2] for x in data_point].index('-1')

        ### Extract entities:
        entity_1 = data_point[0:length_first_entity]
        entity_2 = data_point[(begin_last_entity+1):]
        
        ##
        entity_1_text = ' '.join([x[0] for x in entity_1]).lower()
        entity_2_text = ' '.join([x[0] for x in entity_2]).lower()
                    
        entity_pairs.append((entity_1_text,entity_2_text))

    return entity_pairs
