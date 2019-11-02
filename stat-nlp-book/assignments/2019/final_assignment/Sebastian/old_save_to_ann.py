def old_save_to_ann(data, datadfir):
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