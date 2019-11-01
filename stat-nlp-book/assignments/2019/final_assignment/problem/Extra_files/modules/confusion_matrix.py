# Code for plotting confusion matrix 
# Simply copied from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


def plot_confusion_matrix(y_pred, y_true, n_labels=None,
                          normalize=False):
    """
    This function prints and plots the confusion matrix.
    If n_labels is specified, it only shows the most 'n_labels' most common labels (in y_true)
    Normalization can be applied by setting `normalize=True`.
    """
    
    if normalize:
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Find the most_common
    if n_labels is None:
        x_tick_size = cm.shape[1]
        y_tick_size = cm.shape[0]
        classes = unique_labels(y_true, y_pred)
        label_to_idx = {classes[i]: i for i in range(len(classes))}  # The simple label_to_idx
    else:
        x_tick_size = n_labels
        y_tick_size = n_labels
        most_common = Counter(y_true).most_common(n_labels)
        classes = np.array([tup[0] for tup in most_common])
        
        all_labels = unique_labels(y_true, y_pred)
        label_to_idx = {label: list(all_labels).index(label) for label in classes}
    
    sub_cm = cm[np.array([label_to_idx[label] for label in classes])]
    sub_cm = sub_cm[:, np.array([label_to_idx[label] for label in classes])]
    
    fig, ax = plt.subplots()
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

    fig.tight_layout()
    return ax
