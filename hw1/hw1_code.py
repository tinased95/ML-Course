import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from subprocess import call

def load_data():
    # loading data
    fake_data = np.loadtxt("data/clean_fake.txt", dtype=str, delimiter="\n")
    real_data = np.loadtxt("data/clean_real.txt", dtype=str, delimiter="\n")
    data = np.concatenate((fake_data, real_data), axis=0)   
    
    # Make labels
    real_labels = np.full((len(real_data),1), 'Real')
    fake_labels = np.full((len(fake_data),1), 'Fake')
    labels = np.append(real_labels, fake_labels)

    # initialize the vector
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data)
    
    #split the data to 30% test, 70% training
    data_train, data_validation_test, label_train, label_validation_test = train_test_split(X, labels, test_size=0.3)
    #split the test to 15% validation, 15% test examples
    data_validation, data_test,label_validation, label_test = train_test_split(data_validation_test, label_validation_test, test_size=0.5)

    return data_train,data_validation,data_test,label_train,label_validation,label_test

def select_model():
    training_data,validation_data,testing_data,training_labels,validation_labels,test_labels = load_data()
    max_depth=[1, 4, 8, 12, 16]
    mode = ["gini", "entropy"]

    for m in mode:
        for d in range(len(max_depth)):
            tree_model = tree.DecisionTreeClassifier(max_depth=max_depth[d], criterion=m)
            tree_model.fit(training_data,training_labels)

            #evaluate the performance
            predicted_labels = tree_model.predict(validation_data)
            validation_accuracy = accuracy_score(validation_labels, predicted_labels)
            print('Accuracy of ' +  str(m) + ' with max_depth of', max_depth[d] ,'is:', validation_accuracy*100)

# select_model()

def display_model():
    training_data,validation_data,testing_data,training_labels,validation_labels,test_labels= load_data()
    besttree = DecisionTreeClassifier(criterion = "entropy",random_state = 100,max_depth=50)
    besttree = besttree.fit(training_data,training_labels)
    export_graphviz(besttree,max_depth=2, out_file ='tree.dot',class_names=['Fake','Real'])
    call(["dot", "-Tpng", "tree.dot", "-o tree.png"])

display_model()