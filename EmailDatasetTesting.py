#%%
from cgitb import html
from turtle import numinput
import wheel
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
import mailbox
import csv
import time
from langdetect import detect, DetectorFactory
from nltk.sentiment import SentimentIntensityAnalyzer

t0 = time.time()
bad_mail_data = []
bad_mbox_file = 'phishing3.mbox'
good_mbox_file = 'enron.mbox'
incr = 0
bad_mbox_data = mailbox.mbox(bad_mbox_file)
good_mbox_data = mailbox.mbox(good_mbox_file)

import math
def convertToNumber (s):
    return int.from_bytes(s.encode(), 'little')

def convertFromNumber (n):
    return n.to_bytes(math.ceil(n.bit_length() / 8), 'little').decode()

SIA = SentimentIntensityAnalyzer()
DetectorFactory.seed = 0

def stringToAttributes(content,id,isPhishing):
    # Get URL count
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
    numURLS = len(urls)
    
    # Get IP address count
    ips = re.findall('(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)+',content)
    numIPS = len(ips)

    # Check HTML existence
    html_t = re.findall('(?i)(<HTML>)+',content)
    hasHTML = len(html_t)>0

    # Check Javascript existence
    js_t = re.findall('(?i)(<script)+',content)
    hasJS = len(js_t)>0

    # Check external resource count
    external_t = re.findall('(?i)(href)+',content)
    numEXTERNAL = len(external_t)

    # Get primary language and convert to decimal hash
    lang = detect(content)
    langHash = convertToNumber(lang)/1e5

    # Get compounded VADER score
    textCompound = (SIA.polarity_scores(content)["compound"]+1)/2
    textPos = SIA.polarity_scores(content)["pos"]
    textNeg = SIA.polarity_scores(content)["neg"]
    textNeu = SIA.polarity_scores(content)["neu"]

    return [id,numURLS,numIPS,int(hasHTML),int(hasJS),numEXTERNAL,langHash, textCompound, textPos, textNeg, textNeu, isPhishing]

data = []
email_id = 0

for message in bad_mbox_data:
    incr = incr + 1
    try:
        email_id = email_id + 1
        content = ''
        if message.is_multipart():
            content = ''.join(part.get_payload() for part in message.get_payload())
            bad_mail_data.append(content)
        else:
            #print(message.get_from())
            content = message.get_payload()
            bad_mail_data.append(content)        

        data.append(stringToAttributes(content,email_id,1))
    except:
        pass #

count = len(bad_mail_data)

for root, dirs, files in os.walk("maildir"):
  if files:
    for file in files:
      if count > 0:
        count = count - 1
        f = open(os.path.join(os.path.abspath(root), file))
        content = f.read()
        data.append(stringToAttributes(content,email_id,0))

data = np.array(data)

#Split data into x and y components
x_data = data[:,1:-1]   # Remove ID and answer columns
y_data = data[:,-1]     # Make answer column y_data

# Split data into training,validation and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)
t1 = time.time()
time_preproc_dev = t1-t0

#%%
# % Support Vector Machine Classifier
t2 = time.time()
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay
svmModel = svm.SVC(random_state=42)
svmModel = svmModel.fit(x_train, y_train)
t2a = time.time()
predictions = svmModel.predict(x_test)
print(classification_report(predictions,y_test,digits=3))
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
t3 = time.time()
time_svm_devT = t2a-t2
time_svm_dev = t3-t2

# % Random Forest Classifier
t4 = time.time()
from sklearn.ensemble import RandomForestClassifier
rForestModel = RandomForestClassifier(random_state=42)
rForestModel = rForestModel.fit(x_train,y_train)
t4a = time.time()
predictions = rForestModel.predict(x_test)
print(classification_report(predictions,y_test,digits=3))
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
t5 = time.time()
time_randF_devT = t4a-t4
time_randF_dev = t5-t4

# % Gaussian Naive Bayes
t6 = time.time()
from sklearn.naive_bayes import GaussianNB
gaussNBModel = GaussianNB()
gaussNBModel = gaussNBModel.fit(x_train,y_train)
t6a = time.time()
predictions = gaussNBModel.predict(x_test)
print(classification_report(predictions,y_test,digits=3))
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
t7 = time.time()
time_gaussNB_devT = t6a-t6
time_gaussNB_dev = t7-t6

# % Stocastic Classifier
t8 = time.time()
from sklearn.linear_model import SGDClassifier
sgdModel = SGDClassifier(random_state=42)
sgdModel = sgdModel.fit(x_train,y_train)
t8a = time.time()
predictions = sgdModel.predict(x_test)
print(classification_report(predictions,y_test,digits=3))
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
t9 = time.time()
time_SGD_devT = t8a-t8
time_SGD_dev = t9-t8

# % K-Nearest Neighbours Classifier
t10 = time.time()
from sklearn.neighbors import KNeighborsClassifier
knnModel = KNeighborsClassifier()
knnModel = knnModel.fit(x_train,y_train)
t10a = time.time()
predictions = knnModel.predict(x_test)
print(classification_report(predictions,y_test,digits=3))
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
t11 = time.time()
time_KNN_devT = t10a-t10
time_KNN_dev = t11-t10

# % Decision Tree Classifier
t12 = time.time()
from sklearn.tree import DecisionTreeClassifier
treeModel = DecisionTreeClassifier()
treeModel = treeModel.fit(x_train,y_train)
t12a = time.time()
predictions = treeModel.predict(x_test)
print(classification_report(predictions,y_test,digits=3))
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
t13 = time.time()
time_dTree_devT = t12a-t12
time_dTree_dev = t13-t12

# % Basic NN Classifier
t14 = time.time()
from sklearn.neural_network import MLPClassifier
mlpModel = MLPClassifier(random_state=42,max_iter=500)
mlpModel = mlpModel.fit(x_train,y_train)
t14a = time.time()
predictions = mlpModel.predict(x_test)
print(classification_report(predictions,y_test,digits=3))
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
t15 = time.time()
time_MLP_devT = t14a-t14
time_MLP_dev = t15-t14

# %%
t16 = time.time()

liveData = []

#for root, dirs, files in os.walk("liveTests"):
#  if files:
#    for file in files:
#        f = open(os.path.join(os.path.abspath(root), file))
#        content = f.read()
#        liveData.append(stringToAttributes(content,email_id,0))

goot_test_mbox = 'enron.mbox'
goot_test_mbox_data = mailbox.mbox(goot_test_mbox)


incr2 = 0

for message in goot_test_mbox_data:
  incr2 = incr2 + 1
  if incr2> incr:
    if incr2 > 1500+incr:
      break

    try:
        email_id = email_id + 1
        content = ''
        if message.is_multipart():
            content = ''.join(part.get_payload() for part in message.get_payload())
        else:
            #print(message.get_from())
            content = message.get_payload()   

        liveData.append(stringToAttributes(content,email_id,0))
    except:
        pass #

for root, dirs, files in os.walk("liveTestsBad"):
  if files:
    for file in files:
      try:
        f = open(os.path.join(os.path.abspath(root), file))
        content = f.read()
        liveData.append(stringToAttributes(content,email_id,1))
      except:
        print("Failed to load " + file)

liveData = np.array(liveData)

#Split data into x and y components
x_live_data = liveData[:,1:-1]   # Remove ID and answer columns
y_live_data = liveData[:,-1]     # Make answer column y_data

t17 = time.time()
time_preproc_val = t17-t16

t18 = time.time()
predictions = svmModel.predict(x_live_data)
print("SVM Predicition:")
print(classification_report(predictions,y_live_data,digits=3))
cm = confusion_matrix(y_live_data, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
t19 = time.time()
time_svm_val = t19-t18
predictions = rForestModel.predict(x_live_data)
print("Random Forest Predicition:")
print(classification_report(predictions,y_live_data,digits=3))
cm = confusion_matrix(y_live_data, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
t20 = time.time()
time_rFor_val = t20-t19
predictions = gaussNBModel.predict(x_live_data)
print("Gaussian Predicition:")
print(classification_report(predictions,y_live_data,digits=3))
cm = confusion_matrix(y_live_data, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
t21 = time.time()
time_gaussNB_val = t21-t20
predictions = sgdModel.predict(x_live_data)
print("SGD Predicition:")
print(classification_report(predictions,y_live_data,digits=3))
cm = confusion_matrix(y_live_data, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
t22 = time.time()
time_SGD_val = t22-t21
predictions = knnModel.predict(x_live_data)
print("KNN Predicition:")
print(classification_report(predictions,y_live_data,digits=3))
cm = confusion_matrix(y_live_data, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
t23 = time.time()
time_KNN_val = t23-t22
predictions = treeModel.predict(x_live_data)
print("Decision Tree Predicition:")
print(classification_report(predictions,y_live_data,digits=3))
cm = confusion_matrix(y_live_data, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
t24 = time.time()
time_dTree_val = t24-t23
predictions = mlpModel.predict(x_live_data)
print("CNN Predicition:")
print(classification_report(predictions,y_live_data,digits=3))
cm = confusion_matrix(y_live_data, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
t25 = time.time()
time_mlp_val = t25-t24
# %%
