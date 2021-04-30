import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import sys

# read dataset
snps = pd.read_csv('{0}.input.csv'.format(sys.argv[1]))

train_index = list(pd.read_csv('train_index.csv', header=None)[0])
test_index = list(pd.read_csv('test_index.csv', header=None)[0])
train = snps.iloc[train_index]
test = snps.iloc[test_index]

train_data = train.iloc[:, 6:].values
train_data = train_data.astype(np.float16)
train_labels = train.iloc[:, 5].values
train_labels = train_labels.astype(np.int32)
train_labels = train_labels - 1

test_data = test.iloc[:, 6:].values
test_data = test_data.astype(np.float16)
test_labels = test.iloc[:, 5].values
test_labels = test_labels.astype(np.int32)
test_labels = test_labels - 1

rf_model = RandomForestClassifier()
rf_model.fit(train_data,train_labels)


test_scores = rf_model.predict_proba(test_data)
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(test_labels, test_scores[:,1])
roc_auc = auc(fpr, tpr)
print ('AUC : ', roc_auc)

predict_labels = rf_model.predict(test_data)
cm = confusion_matrix(test_labels, predict_labels)
print('Confusion Matrix : \n', cm)

total = sum(sum(cm))
accuracy =(cm[0,0]+cm[1,1])/total
print ('Accuracy : ', accuracy)

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity)

specificity = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity)
