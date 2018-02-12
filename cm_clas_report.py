import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np

def classification_report_df(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = list(filter(None, line.split(' ')))
        print(row_data)
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    # avg line
    str_list = lines[-2].split(' ')
    row_data = list(filter(None, str_list)) # fastest
    row = {}
    row['class'] = row_data[0]+row_data[1]+row_data[2]
    row['precision'] = float(row_data[3])
    row['recall'] = float(row_data[4])
    row['f1_score'] = float(row_data[5])
    row['support'] = float(row_data[6])
    report_data.append(row)
    # build final df
    df_report = pd.DataFrame.from_dict(report_data)
    df_latex = df_report.to_latex()
    df_latex = df_latex.replace('\n\\toprule', '')  # erase top rule, mid rule and bottom rule line
    df_latex = df_latex.replace('\n\\midrule', '')  # erase top rule, mid rule and bottom rule line
    df_latex = df_latex.replace('\n\\bottomrule', '')  # erase top rule, mid rule and bottom rule line
    return df_latex


# Plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    if normalize:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    else:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
