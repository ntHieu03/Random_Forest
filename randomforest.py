# -*- coding: utf-8 -*-
"""
Created on Sat May 13 14:21:46 2023

@author: pc013
"""

import pandas as pd
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import random
import math
import matplotlib.pyplot as plt

def stratify_sampling(df):
    X = df.iloc[:, df.columns != 'class']
    y = df[['class']]

    #acc_DT = []
    acc_RF = []

    for i in range(100):
        Train_x, Test_x, Train_y, Test_y = train_test_split(X, y, stratify=y, test_size=0.4, random_state=random.randint(0, 100000))

        #model = DecisionTreeClassifier()
        #model.fit(Train_x, Train_y)
        #predictions = model.predict(Test_x)
        #.append(metrics.accuracy_score(Test_y, predictions))
        #print("Accuracy Decision Tree:", metrics.accuracy_score(Test_y, predictions))

        rf_model = RandomForestClassifier(n_estimators=100, max_features=int(math.sqrt(X.shape[1])) + 1)
        rf_model.fit(Train_x, Train_y.values.ravel())
        pred_y = rf_model.predict(Test_x)
        acc_RF.append(metrics.accuracy_score(Test_y, pred_y))
        print("Accuracy Random Forest:", metrics.accuracy_score(Test_y, pred_y))

    #print("Average Accuracy Decision Tree:", sum(acc_DT) / len(acc_DT))
    print("Average Accuracy Random Forest:", sum(acc_RF) / len(acc_RF))

    results = [acc_RF]
    names = ['Random Forest']
    #results = [acc_DT]
    #names = ['Decision Tree']
    
    plt.boxplot(results, labels=names)
    plt.ylabel('Accuracy')

    plt.show()
    

def main():
    df = pd.read_csv('C:\\Users\\pc013\\Downloads\\wine.csv')
    df.columns.values[0] = "class"
    stratify_sampling(df)
    num_classes = len(df['class'].unique())
    print("Number of classes:", num_classes)

if __name__ == "__main__":
    main()
