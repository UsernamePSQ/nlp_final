from Extra_files.modules.Entity_list_identifyer import List_identifyer, Abr_identifyer
import numpy as np

def _add_rules(data_m_XY, data_raw):
    '''
    So far this function:

    1) Forces abbrevations to be synonyms with the entity right in front of
    2) Links lists, so none are Synonyms, if any is Hyponym or Hyponym-reverted, it marks them all this
    '''

    ## Add rules sequentially
    data_m_XY = _add_strong_abr_rules(data_m_XY, data_raw)
    data_m_XY = _add_weak_list_rules(data_m_XY, data_raw)

    return data_m_XY

def _add_strong_abr_rules(data_m_XY, data_raw):
    '''
    This forces abbrevations to be synonyms with the entity right in front of
    '''
    # Get all abbreviations
    all_abrs = Abr_identifyer().find_abbrevations(data_raw)

    predictions = data_m_XY['data_Y']

    ## Change the predictions
    for idx in range(len(predictions)):

        # Extract relevant data from data_point
        txt, entity_1, entity_2 = data_m_XY['metadata'][idx]
        abrs = all_abrs[txt]

        # Check if the entities are in abbreviations
        if any([(entity_1, entity_2) == syn for syn in abrs]) or any([(entity_2, entity_1) == syn for syn in abrs]):
            predictions[idx] = "Synonym"
        
    data_m_XY['data_Y'] = predictions

    return data_m_XY


def _add_weak_list_rules(data_m_XY, data_raw):
    '''
    This links lists so following are met:
    
    1) Nothing in lists are allowed to be Synonym
    2) If one is Hyponym xor Hyponym-reverted, all are
    3) If both Hyponym and Hyponym-reverted are present, chooses randomly (for now, maybe percentages later)
    '''

    ## Get lists and abbreviations
    all_linked_lists = List_identifyer().find_lists(data_raw)

    
    ## Create index locations for faster lookups
    all_txt_files = np.array(list(all_linked_lists.keys()))
    txt_files = [meta[0] for meta in data_m_XY['metadata']]

    txt_indices = {file: np.argwhere(np.array(txt_files) == file).reshape(-1) for file in all_txt_files}
    Falsely_removed_synonyms = 0
    Falsely_removed_rels = 0
    counter_not_altered = 0
    counter_altered = 0
    #Only loop over txt_files with links
    for txt in all_txt_files:

        #Extract the X's and Y's with the file as metadata
        txt_X = [data_m_XY['data_X'][idx] for idx in txt_indices[txt]]
        txt_Y = [data_m_XY['data_Y'][idx] for idx in txt_indices[txt]]
        entities = [tuple(data_m_XY['metadata'][idx][1:]) for idx in txt_indices[txt]]
        
        #Extract links from that txt
        linked_lists = all_linked_lists[txt]

        for links in linked_lists:

            ## Find the data with entities from the list
            idxs_w_entities = [idx for idx in range(len(entities)) if \
                               len(np.intersect1d(np.array(links), np.array(entities[idx]))) >= 1]

            ## No synonyms with an entity from a list
            for idx in idxs_w_entities:
                if txt_Y[idx] == 'Synonym':
                    Falsely_removed_synonyms += 1
                    txt_Y[idx] = 'NONE'
                    data_m_XY['data_Y'][txt_indices[txt][idx]] = 'NONE'
            
            ## No relations between them
            for idx in idxs_w_entities:
                if entities[idx][0] in links and entities[idx][1] in links:
                    if txt_Y[idx] != 'NONE':
                        Falsely_removed_rels += 1
                    txt_Y[idx] = 'NONE'
                    data_m_XY['data_Y'][txt_indices[txt][idx]] = 'NONE'
            
            ## If one is Hyponym or Hyponym_reverted, let them all damn be.
            for idx in idxs_w_entities:
                
                if txt_Y[idx] == 'NONE':
                    continue
                elif txt_Y[idx] == 'Hyponym':
                    relation = 'Hyponym'
                else:
                    relation = 'Hyponym_reverted'

                if entities[idx][0] in links:
                    other_idxs = [entities.index((ent,entities[idx][1])) for ent in links]
                    for id in other_idxs:

                        if data_m_XY['data_Y'][txt_indices[txt][id]] == relation:
                            counter_not_altered += 1 * (id != idx)
                        else:
                            data_m_XY['data_Y'][txt_indices[txt][id]] = relation
                            counter_altered += 1
                            
                elif entities[idx][1] in links:
                    other_idxs = [entities.index((entities[idx][0], ent)) for ent in links]
                    for id in other_idxs:
                        if data_m_XY['data_Y'][txt_indices[txt][id]] == relation:
                            counter_not_altered += 1 * (id != idx)
                        else:
                            data_m_XY['data_Y'][txt_indices[txt][id]] = relation
                            counter_altered += 1
                else:
                    raise Exception("This should not happen!")
    print("Number of removed synonyms: {}".format(Falsely_removed_synonyms))
    print("Number of removed hyponym-relations within list: {}".format(Falsely_removed_rels))
    print("Number of labels changed to hyponyms: {}".format(counter_altered))
    print("Number of labels that would be changed but already in list of hyponyms: {}".format(counter_not_altered))
    return data_m_XY



