# Code for plotting confusion matrix 
# Simply copied from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib as mpl
import importlib


def get_dataframe(data, data_m_XY, y_true, y_base, y_weak):
    '''
    This functions creates a dataframe for later analysis
    '''

    '''
    # Mask only where they are not all NONE
    mask = (y_base != 'NONE') | (y_weak != 'NONE') | (y_true != 'NONE')
    indices = np.arange(len(data_m_XY['data_Y']))[mask]

    #Subset
    y_base = y_base[mask]
    y_weak = y_weak[mask]
    y_true = y_true[mask]
    metadata = np.array(data_m_XY['metadata'])[mask]
    '''
    metadata = np.array(data_m_XY['metadata'])

    

    #Create df
    df_err_an = pd.DataFrame({'file' : [],
                            'Sentence between entities': [],
                            'Entity 1': [],
                            'Entity 2': []})

    ## Append to df
    for idx in range(len(y_true)):
        txt, T1, T2 = metadata[idx]
        B_mask = np.array([x[0] == 'B' for x in data[txt]['IOBtags']])
        B_indices = np.arange(len(data[txt]['IOBtags']))[B_mask]
        assert len(B_indices) == len(data[txt]['annotation_names']) #Check i found all annotations

        ## Find first entity (from and to)
        index_start = B_indices[data[txt]['annotation_names'].index(T1)]
        rest_iob_tags = np.array(data[txt]['IOBtags'])[(index_start+1):]
        mask = [(x[0] in ['B','O']) for x in rest_iob_tags]
        index_for_next = np.arange(len(rest_iob_tags))[mask][0]
        entity_1_end = index_start + index_for_next
        
        ## Find second entity
        index_ent2_begin = B_indices[data[txt]['annotation_names'].index(T2)]
        if index_ent2_begin == len(data[txt]['tokens']) - 1:
            index_end = index_ent2_begin
        else:
            rest_iob_tags = np.array(data[txt]['IOBtags'])[(index_ent2_begin+1):]
            mask = [(x[0] in ['B','O']) for x in rest_iob_tags]
            index_for_next = np.arange(len(rest_iob_tags))[mask][0]
            index_end = index_ent2_begin + index_for_next
            
        assert index_end == index_start + len(data_m_XY['data_X'][idx])-1 #Check i get same index_end as Skjøtt

        #Create entity_1
        entity_1 = data[txt]['tokens'][index_start]
        for i in range(index_start+1,entity_1_end+1):
            if data[txt]['locations'][i-1][1] == data[txt]['locations'][i][0]:
                entity_1 += data[txt]['tokens'][i]
            elif data[txt]['locations'][i-1][1] == data[txt]['locations'][i][0] - 1:
                entity_1 += " " + data[txt]['tokens'][i]  
                
        #Create entity_2
        entity_2 = data[txt]['tokens'][index_ent2_begin]    
        for i in range(index_ent2_begin+1,index_end+1):
            if data[txt]['locations'][i-1][1] == data[txt]['locations'][i][0]:
                entity_2 += data[txt]['tokens'][i]
            elif data[txt]['locations'][i-1][1] == data[txt]['locations'][i][0] - 1:
                entity_2 += " " + data[txt]['tokens'][i]
            
        #Create sentence
        sentence = data[txt]['tokens'][index_start]
        for i in range(index_start+1, index_end+1):
            if data[txt]['locations'][i-1][1] == data[txt]['locations'][i][0]:
                sentence += data[txt]['tokens'][i]
            else:
                sentence += " " + data[txt]['tokens'][i]
        
        df_err_an = df_err_an.append({'file':txt,
                                    'Sentence between entities':sentence,
                                    'Entity 1':entity_1,
                                    'Entity 2':entity_2},ignore_index=True)
    #Append the y's
    df_err_an['True label'] = y_true
    df_err_an['Base label'] = y_base
    df_err_an['Weak learning label'] = y_weak

    return df_err_an


def plot_correct_labels(df_err_an):
    importlib.reload(sns)
    pd.options.mode.chained_assignment = None  # Remove .loc-warnings from pandas

    # Create column indicating the wrong model
    df_correct_labels = df_err_an

    df_correct_labels['Model'] = 'Both'
    df_correct_labels.loc[(df_correct_labels['True label'] == df_correct_labels['Base label']) & \
                        (df_correct_labels['True label'] != df_correct_labels['Weak learning label']) ,'Model'] = 'Base model'
    df_correct_labels.loc[(df_correct_labels['True label'] == df_correct_labels['Weak learning label']) & \
                        (df_correct_labels['True label'] != df_correct_labels['Base label']) ,'Model'] = 'Weak model'
    df_correct_labels.loc[(df_correct_labels['True label'] != df_correct_labels['Weak learning label']) & \
                        (df_correct_labels['True label'] != df_correct_labels['Base label']) ,'Model'] = 'None'

    warnings.filterwarnings('ignore') #Seaborn gets warning, but not a problem

    df = df_correct_labels
    tmp = df["True label"].groupby(df["Model"]).value_counts(normalize=False).rename("Number of correct labels").reset_index()

    ## Add empty row if doesn't exist
    labels = ['NONE','Synonym', 'Hyponym', 'Hyponym_reverted']
    models = ['None', 'Both', 'Base model', 'Weak model']

    for label in labels:
        for model in models:
            if sum((tmp['True label'] == label) & (tmp['Model'] == model)) == 0:
                tmp = tmp.append({'True label': label,
                                 'Model': model,
                                 'Number of correct labels': 0},
                                 ignore_index = True)

    #Add 'both' to the individual
    for label in labels:
        for model in ['Base model', 'Weak model']:
            tmp.loc[(tmp['True label'] == label) & (tmp['Model'] == model),'Number of correct labels'] += \
                np.array(tmp.loc[(tmp['True label'] == label) & (tmp['Model'] == 'Both'),'Number of correct labels'])

    # Make into percentage
    for label in labels:
        tmp.loc[tmp['True label'] == label,'Number of correct labels'] /= sum(df_correct_labels['True label'] == label)

    ax = sns.barplot(x="True label", y="Number of correct labels", hue="Model", data=tmp)
    ax.set_ylabel('Percentage correct labels')
    ax.set_title('Percentage of correct labels within each category',fontdict = {'fontsize':  15})
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    warnings.filterwarnings('once') #Turn warning on again

    return ax

def plot_confusion_matrix(y_true, y_pred, n_labels = None,
                          normalize=False,ax = None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    sns.reset_orig()
    if normalize:
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    #Find the most_common
    if n_labels is None:
        x_tick_size = cm.shape[1]
        y_tick_size = cm.shape[0]
        classes = unique_labels(y_true, y_pred)
        label_to_idx = {classes[i]: i for i in range(len(classes))} #The simple label_to_idx
    else:
        x_tick_size = n_labels
        y_tick_size = n_labels
        most_common = Counter(y_true).most_common(n_labels)
        classes = np.array([tup[0] for tup in most_common])
        
        all_labels = unique_labels(y_true, y_pred)
        label_to_idx = {label:list(all_labels).index(label) for label in classes}
    
    sub_cm = cm[np.array([label_to_idx[label] for label in classes])]
    sub_cm = sub_cm[:,np.array([label_to_idx[label] for label in classes])]

    if ax is None:
        ax = plt.gca()
    im = ax.imshow(sub_cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(x_tick_size),
           yticks=np.arange(y_tick_size),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
        
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = sub_cm.max() / 2.
    
    for i in range(x_tick_size):
        for j in range(y_tick_size):               
            ax.text(j, i, format(sub_cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if sub_cm[i, j] > thresh else "black")

    return ax