#..............................................Prediction.............................................#

import os
import time
import joblib
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,f1_score
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

clf = joblib.load('BEST_Model.pkl')

def Prediction(file):
    df = pd.read_csv(file)
    X= df.values
    label = clf.predict(X)
    print(label)
    return label

'''
def main():
    path = 'LoadForecasting_Test'
    while True:
        for(direcpath,direcnames,files) in os.walk(path):
            for file in files:
                if 'csv' in file:
                    val = (Prediction(file))
                    print('Load Forecasting Under Cyberattacks:')
                    print('\n')
                    print('Detected:',val)
                    f = open("Cyberattacks.txt",'a')
                    print(val, file = f)
                    f.close()                        
                    time.sleep(1)    
                    os.remove(file)
    
main()'''
#Prediction('LoadForecasting_Test/scenario3.csv')
