from Extra_files.modules.Entity_list_identifyer import List_identifyer, Abr_identifyer
import numpy as np

def _sebastians_dummy_model(data_dict,data_raw):
    """
    This creates two types of labels:

    1) Abbreviations are labeled as synonyms.
    2) Lists are all labeled as Hyponym of the entity right in front of the list.
    """

    ## Get lists and abbreviations
    all_linked_lists = List_identifyer().find_lists(data_raw)
    all_abrs = Abr_identifyer().find_abbrevations(data_raw)

    predictions = []
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
