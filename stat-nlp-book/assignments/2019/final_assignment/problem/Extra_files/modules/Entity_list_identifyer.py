from collections import defaultdict


class Abr_identifyer():
    '''
    This class identifies, when there are abrevations. These are defined as either

    1) xxxx (YYY)
    2) xxxx (YYYs)

    '''

    def __init__(self):
        self.entity_lists = defaultdict(list)
        self.tmp = []

    def find_abbrevations(self,data):

        for txt, data_point in data.items():

            self.txt = txt
            tokens = data_point['tokens']
            IOBtags = data_point['IOBtags']
            ann_names = data_point['annotation_names']

            token_idx = 0
            entity_number = 0

            #We will iterate over tokens using token_idx and search for lists.
            while True:

                #When we reach this token_idx, we will stop (later code explains)
                if token_idx >= len(tokens)-2:
                    break

                #If the current is 'B', add to tmp
                if IOBtags[token_idx][0] == 'B':
                    self.tmp = [ann_names[entity_number]]
                    entity_number += 1
                    token_idx += 1
                    continue

                # If this is I, skip
                if IOBtags[token_idx][0] == 'I':
                    token_idx += 1
                    continue  # Next loop
                
                #If the current is neither B- nor I-:

                #We need '(', 'B-', ')'
                req_1 = tokens[token_idx] == '(' and IOBtags[token_idx+1][0] == 'B' and tokens[token_idx+2] == ')'
                
                #We need the middle one to have only capitals or capitals with s in the end
                paranthesis_token = tokens[token_idx+1][0]

                req_2_option_1 = all(map(str.isupper, paranthesis_token))
                req_2_option_2 = all(map(str.isupper, paranthesis_token[:-1])) and paranthesis_token[-1] == 's'
                
                if req_1 and (req_2_option_1 or req_2_option_2):
                    self.tmp.append(ann_names[entity_number])
                    assert len(self.tmp) <=2
                    
                    # Add if this is second
                    if len(self.tmp) == 2:
                        self.entity_lists[self.txt].append(tuple(self.tmp))

                    entity_number += 1
                    token_idx += 3  # Jump til after the parenthesis 
                    self.tmp = []  # Reset tmp
                else:
                    token_idx += 1
                    self.tmp = []  # Reset tmp

                # End if
            # End while loop
        # End loop over data_points

        return self.entity_lists


class List_identifyer():
    '''
    This class identifies, when entities are named iteratively in a list.

    The following five types are defined as lists of entities:

    XXXX, XXXX, XXXX (at least three entities with only comma in between)
    XXXX, XXXX and XXXX (at least two entities with comma, and the last with 'and')
    XXXX, XXXX, and XXXX (at least two entities with comma, and the last with ', and')
    XXXX, XXXX or XXXX (at least two entities with comma, and the last with 'or')
    XXXX, XXXX, or XXXX (at least two entities with comma, and the last with ', or')
    '''
    
    def __init__(self):
        self.entity_lists = defaultdict(list)
        self.tmp = []  # Used in 'find_lists

    def find_lists(self, data):

        for txt, data_point in data.items():
            self.txt = txt
            tokens = data_point['tokens']
            IOBtags = data_point['IOBtags']
            ann_names = data_point['annotation_names']

            token_idx = 0
            entity_number = 0

            #We will iterate over tokens using token_idx and search for lists.
            while True:

                #When we reach this token_idx, we will stop (later code explains)
                if token_idx >= len(tokens)-1:
                    break

                #If the current is 'B', add to tmp
                if IOBtags[token_idx][0] == 'B':
                    self._add_list_and_reset()  # Add tmp and reset
                    self.tmp = [ann_names[entity_number]]
                    entity_number += 1
                    token_idx += 1
                    continue

                # If the next is I, skip                    
                if IOBtags[token_idx][0] == 'I':
                    token_idx += 1
                    continue  # Next loop
                
                # If the next is not I:

                # Check options for adding items to list
                option_1 = tokens[token_idx] == ',' and IOBtags[token_idx+1][0] == 'B'

                option_2 = tokens[token_idx] == 'and' and IOBtags[token_idx+1][0] == 'B'
                option_3 = tokens[token_idx] == 'or' and IOBtags[token_idx+1][0] == 'B'

                if token_idx + 2 < len(tokens):
                    option_4 = tokens[token_idx] == ',' and tokens[token_idx+1] == 'and' and IOBtags[token_idx+2][0] == 'B'
                    option_5 = tokens[token_idx] == ',' and tokens[token_idx+1] == 'or'  and IOBtags[token_idx+2][0] == 'B'
                else:
                    option_4 = False
                    option_5 = False


                #Check if we should add to list or not
                if option_1:
                    self.tmp.append(ann_names[entity_number])
                    entity_number += 1
                    token_idx += 2  # Jump to right after B.
                elif option_2 or option_3:
                    self.tmp.append(ann_names[entity_number])
                    entity_number += 1
                    token_idx += 2  #Jump to right after B.
                    self._add_list_and_reset()  # Add tmp and reset
                elif option_4 or option_5:
                    self.tmp.append(ann_names[entity_number])
                    entity_number += 1
                    token_idx += 3  # Jump to right after B.       
                    # Add tmp and reset
                    self._add_list_and_reset()    
                else:
                    token_idx += 1
                    self._add_list_and_reset()

                # End if
            # End while loop
        # End loop over data_points

        return self.entity_lists

    def _add_list_and_reset(self):

        #If length is minimum 3, add it
        if len(self.tmp) >= 3:
            self.entity_lists[self.txt].append(self.tmp)
        
        #Reset tmp
        self.tmp = []

        return

        