from statnlpbook.scienceie import print_report
from sklearn.metrics import precision_recall_fscore_support


def f1_score_ala_calc_measures(y_true, y_pred, printing = True):
    targets = ["Hyponym", "Synonym", "Hypernym"]
    prec, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=targets, average=None)
    # unpack the precision, recall, f1 and support
    metrics = {}
    for k, target in enumerate(targets):
        metrics[target] = {
            'precision': prec[k],
            'recall': recall[k],
            'f1-score': f1[k],
            'support': support[k]
        }

    prec, recall, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=targets, average='micro')
        
    metrics['overall'] = {
        'precision': prec,
        'recall': recall,
        'f1-score': f1,
        'support': sum(support)}

    if printing:
        print_report(metrics, targets)

    return metrics['overall']['f1_score']






