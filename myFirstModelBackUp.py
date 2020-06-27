# -*- coding: utf-8 -*-
"""
MURAT KARAKAYA AKADEMİ

NASIL KODLASAM?

"""
# %%
#LOAD DATASET

from sklearn.datasets import load_iris
dataSet = load_iris()

features = dataSet.data
labels = dataSet.target
labelsNames = list(dataSet.target_names)
featureNames = dataSet.feature_names

print([labelsNames[i] for i in labels[47:52]])
print(featureNames)




# %%
# ANALYZE DATA
import pandas as pd

#print(type(features))

featuresDF= pd.DataFrame(features)
featuresDF.columns = featureNames

#print(type(featuresDF))
#print(featuresDF.describe())
print(featuresDF.info())


# %%
# VISUALIZE DATA

featuresDF.plot(x="sepal length (cm)", y= "sepal width (cm)", kind= "scatter")

# %%
# SELECT MODEL

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=8)


# %%
# SPLIT DATASET

from sklearn.model_selection import train_test_split

X = features
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# %%
# TRAIN MODEL

clf.fit(X_train, y_train)
accuracy = clf.score(X_train, y_train)
print("accuracy on train data {:.2}%".format(accuracy))

# %%
# TEST MODEL

accuracy = clf.score(X_test,y_test)
print("accuracy on test data {:.2}%".format(accuracy))


# %%
# IMPROVE

# %%
# SAVE MODEL
from joblib import dump, load
filename="myFirstSavedModel.joblib"
dump(clf, filename)

# %%
# LOAD MODEL

clfUploaded = load(filename)

# %%
# TEST WITH UPLOADED MODEL

accuracy = clfUploaded.score(X_test,y_test)
print("accuracy on test data {:.2}%".format(accuracy))

