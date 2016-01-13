#!/usr/bin/python

""" 
    SVM mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################

#clf = svm.SVC(kernel="linear")
clf = svm.SVC(kernel="rbf", C=10000.) #higher C means good accuracy, more data fitting
t0 = time()
#reducing the length of training data for faster calculation
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t1 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"
ans1 = pred[10]
print "Prediction by SVM for element 10", (ans1)
ans2 = pred[26]
print "Prediction by SVM for element 26", (ans2)
ans3 = pred[50]
print "Prediction by SVM for element 50", (ans3)
count = 0
for num in range(1,1700) :
    if pred[num] == 1:
        count+=1
print "Out of 1700 test cases, ones predicted belonging to class 1 are:", count
accuracy = accuracy_score(labels_test, pred)
print(accuracy)

#########################################################


