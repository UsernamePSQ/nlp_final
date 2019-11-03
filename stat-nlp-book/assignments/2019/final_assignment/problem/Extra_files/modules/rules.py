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
        txt, entity_1, entity_2 = data_data_m_XYdict['metadata'][idx]
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

    txt_indices = [np.where(txt_files == file) for file in all_txt_files]

    for txt in txt_files:
        # for links in link
        # Extract the indices.. again
            # the function (on the correct indices)

    for idx in range(len(data_dict['data_X'])):

        # Extract relevant data from data_point
        txt, entity_1, entity_2 = data_dict['metadata'][idx]
        links = all_linked_lists[txt]
        abrs = all_abrs[txt]

        # Link the lists to the entity right before
        rb = [data_raw[txt]['annotation_names'][data_raw[txt]['annotation_names'].index(link[0])-1] for link in links]

        # Check if the entities are in abreviations
        if any([(entity_1, entity_2) == syn for syn in abrs]) or any([(entity_2, entity_1) == syn for syn in abrs]):
            predictions.append("Synonym")
        # Check if entities are in hyponyms
        elif any([((entity_1 == rb[idx]) and (entity_2 in links[idx])) for idx in range(len(links))]):
            predictions.append("Hyponym_reverted")
        # Dette burde aldrig ske:
        elif any([((entity_2 == rb[idx]) and (entity_1 in links[idx])) for idx in range(len(links))]):
            raise Exception("This should never happen, because the data-set \
                            is ordered, and we only let lists be hypernym of previous word")
            predictions.append("Hyponym")

        # Else none
        else:
            predictions.append("NONE")


def _sebastians_dummy_model(data_m_XY,data_raw):
    """
    This creates two types of labels:

    1) Abbreviations are labeled as synonyms.
    2) Lists are all labeled as Hyponym of the entity right in front of the list.
    """

    ## Get lists and abbreviations
    all_linked_lists = List_identifyer().find_lists(data_raw)
    all_abrs = Abr_identifyer().find_abbrevations(data_raw)

    # Create hash tables for faster lookup in looping:


    ## Create the predictions
    for idx in range(len(data_dict['data_X'])):

        # Extract relevant data from data_point
        txt, entity_1, entity_2 = data_dict['metadata'][idx]
        links = all_linked_lists[txt]
        abrs = all_abrs[txt]

        # Link the lists to the entity right before
        rb = [data_raw[txt]['annotation_names'][data_raw[txt]['annotation_names'].index(link[0])-1] for link in links]

        # Check if the entities are in abreviations
        if any([(entity_1, entity_2) == syn for syn in abrs]) or any([(entity_2, entity_1) == syn for syn in abrs]):
            predictions.append("Synonym")
        # Check if entities are in hyponyms
        elif any([((entity_1 == rb[idx]) and (entity_2 in links[idx])) for idx in range(len(links))]):
            predictions.append("Hyponym_reverted")
        # Dette burde aldrig ske:
        elif any([((entity_2 == rb[idx]) and (entity_1 in links[idx])) for idx in range(len(links))]):
            raise Exception("This should never happen, because the data-set \
                            is ordered, and we only let lists be hypernym of previous word")
            predictions.append("Hyponym")

        # Else none
        else:
            predictions.append("NONE")

    return np.array(predictions)
