# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 09:35:47 2019

@author: enigmarikki
"""
import re
import pandas as pd
from nltk import PorterStemmer
from nltk.corpus import stopwords
import nltk
import numpy as np
import scipy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
ohe = OneHotEncoder()
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorise = TfidfVectorizer()

dataset = pd.read_csv('Train.csv')

testdata = pd.read_csv('Test.csv')

def cleanSummary(string):
    text_data = re.sub('[^a-zA-Z]' , '  ' ,string)
    #Stemming the Words
    ps = PorterStemmer()
    #Tokenizing the Words
    text_data = nltk.word_tokenize(text_data)
    lmt = WordNetLemmatizer()
    #Removing stop-words
    text_data = [ ps.stem(i) for i in text_data if not i in set(stopwords.words())]
    text_data = [lmt.lemmatize(i) for i in text_data]
    text_data = ' '.join(text_data)
    text_data = text_data.lower()
    return text_data

#cleaning the Vendor_Code column
vendorCode = dataset['Vendor_Code']
vCtest = testdata['Vendor_Code']
vendorCodeEncoded = [i.split('-')[-1] for i in vendorCode]
vCtestEncoded = [i.split('-')[-1] for i in vCtest]
#cleaning the GL-code
glCode = dataset.iloc[: , 2]
glCodetest = testdata.iloc[: , 2]
labelEncodedGC = le.fit(glCode)
labeltestEncodedGC = labelEncodedGC.transform(glCodetest)
glCodeEncoded = ohe.fit_transform(labelEncodedGC.transform(glCode).reshape(len(glCode),1)).toarray()
glCodeTestEncoded = ohe.fit_transform(labeltestEncodedGC.reshape(len(glCodetest),1)).toarray()
#selecting inv_amt
invAmt = dataset.iloc[: , 3]
invTestAmt = testdata.iloc[: , 3]
#Cleaning the Description
descrpitionLst = dataset.iloc[:, 4]
descriptionTest = testdata.iloc[ : , 4]
descriptor = []
descriptorTest =[]
print('starting to clean the summaries!')
c =0
for i in descrpitionLst:
    print(c , 'done..')
    c = c+1
    descriptor.append(cleanSummary(i))
c =0
for i in descriptionTest:
    print(c, 'done..')
    c = c+1
    descriptorTest.append(cleanSummary(i))
#Vectorising the Descriptions
print('starting to vectorise the features')
vec = vectorise.fit(descriptor)
vecTrain = vec.transform(descriptor).toarray()
vecTest = vec.transform(descriptorTest).toarray()
print ('vectorising done..')
arr = np.column_stack(
            (vendorCodeEncoded,
             glCodeEncoded,
             invAmt,
             #labelEncodeD1,
             vecTrain))
arr1 = np.column_stack(
            (vCtestEncoded,
             glCodeTestEncoded,
             invTestAmt,
             #labelEncodeDT1,
             vecTest
                    ))

xtrain ,xtest  = arr,arr1  
#Cleaning Y
y = dataset.iloc[:, 5]
Y = le.fit(y)
y = Y.transform(y)

#putting the input parameters together and Scaling the data
from sklearn.preprocessing import StandardScaler 
ss = StandardScaler()
x = ss.fit_transform(xtrain)
xtest = ss.fit_transform(xtest)
#Fitting the RandomForestClassifier model and testing for accuracy
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
X_train , X_test , Y_train, Y_test = model_selection.train_test_split(x , y ,test_size =0.2)

classifierRFC = RandomForestClassifier()
classifierRFC.fit(X_train , Y_train)
ypred = classifierRFC.predict(X_test)
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(Y_test , ypred)
print(cm1)
print(accuracy_score(Y_test, ypred))

#classify for test.csv
classify = classifierRFC.predict(xtest)
lst = list(Y.inverse_transform(classify))
df = pd.DataFrame()
df['Inv_Id'] = testdata.iloc[: ,0]
df['Product_Category'] = lst

df.to_csv('RFC.csv')
