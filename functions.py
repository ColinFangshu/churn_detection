# import basic libraries
import pandas as pd
import numpy as np
np.random.seed(0)
import itertools

# import the functions.py
import functions as func

# import libraries for visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# import sklearn libraries for preprocessing, pipeline, model selection, model evaluation
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score, confusion_matrix, auc, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# import logisticRegression model
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# import RandomForest model
from sklearn.ensemble import RandomForestClassifier

# import XGBoost model
import xgboost as xgb

# import SupervisedVectorMachine model
from sklearn import svm

from sklearn.preprocessing import label_binarize

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=cm.Blues):
    """ Return the plot of the confusion matrix"""


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    thresh = cm.max() / 2.
    # print('The threshold is: {}'.format(thresh))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_roc_curve(y_test, y_score, title='Receiver operating characteristic (ROC) Curve'):
    """ Return the plot of the roc curve"""


    fpr, tpr, _ = roc_curve(y_test, y_score)
    print('AUC: {}'.format(auc(fpr, tpr)))
    #Seaborns Beautiful Styling
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    plt.figure(figsize=(10,8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def precision(cm):
    """ Return the precision of the confusion matrix"""


    return cm[1,1]/(cm[1,1]+cm[0,1])

def accuracy(cm):
    """ Return the accuracy of the confusion matrix"""


    return (cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])

def recall(cm):
    """ Return the recall of the confusion matrix"""


    return cm[1,1]/(cm[1,1]+cm[1,0])

def specificity(cm):
    """ Return the specificity of the confusion matrix"""


    return cm[0,0]/(cm[0,1]+cm[0,0])

def F1(cm):
    """ Return the F1 score of the confusion matrix"""


    return 2*(precision(cm)*recall(cm))/(precision(cm)+recall(cm))

def plot_roc_curve_RF(y_test, y_score):
    """ Return the plot of the roc curve"""

    n_classes = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print('AUC: {}'.format(auc(fpr, tpr)))
    #Seaborns Beautiful Styling
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    plt.figure(figsize=(10,8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# def plot_all_roc_curve(y_test, list_of_y_score, label, title='Receiver operating characteristic (ROC) Curve'):
#     """ Return the plot of the all the roc curve"""

#     fpr = []
#     tpr = []
#     for idx,y_score in enumerate(list_of_y_score):
#         fpr[:,idx], tpr[:,idx], _ = roc_curve(y_test, y_score)
#     #Seaborns Beautiful Styling
#     sns.set_style("darkgrid", {"axes.facecolor": ".9"})

#     fig, ax = plt.subplots(figsize=(10,8))
#     # plt.figure(figsize=(10,8))
#     lw = 2
#     ax = plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label=label[0])
#     ax = plt.plot(fpr, tpr, color='darkred',
#          lw=lw, label=label[1])
#     ax = plt.plot(fpr, tpr, color='darkyellow',
#          lw=lw, label=label[2])
#     ax = plt.plot(fpr, tpr, color='darkblue',
#          lw=lw, label=label[3])
#     ax = plt.plot(fpr, tpr, color='darkgreen',
#          lw=lw, label=label[4])
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.yticks([i/20.0 for i in range(21)])
#     plt.xticks([i/20.0 for i in range(21)])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(title)
#     plt.legend(loc="lower right")
#     plt.show()

# def plot_roc_curve(C_param_range, X_train_resampled, y_train_resampled, y_test, y_score):

#     names = C_param_range
#     colors = sns.color_palette("Set2", n_colors=len(names))

#     plt.figure(figsize=(10,8))

#     for n, c in enumerate(C_param_range):
#         #Fit a model
#         logreg = LogisticRegression(fit_intercept = False, C = c) #Starter code
#         model_log = logreg.fit(X_train_resampled, y_train_resampled)
#         print(model_log) #Preview model params

#         #Predict
#         y_hat_test = logreg.predict(X_test)

#         y_score = logreg.fit(X_train_resampled, y_train_resampled).decision_function(X_test)

#         fpr, tpr, thresholds = roc_curve(y_test, y_score)
        
#         print('AUC for {}: {}'.format(names[n], auc(fpr, tpr)))
#         lw = 2
#         plt.plot(fpr, tpr, color=colors[n],
#                 lw=lw, label='ROC curve Regularization Weight: {}'.format(names[n]))
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])

#     plt.yticks([i/20.0 for i in range(21)])
#     plt.xticks([i/20.0 for i in range(21)])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic (ROC) Curve')
#     plt.legend(loc="lower right")
#     plt.show()