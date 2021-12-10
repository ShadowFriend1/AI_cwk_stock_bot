#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:19:03 2021

@author: jack
"""

from csv import writer, reader
import pandas as pd
import matplotlib.pyplot as plt
import Regression_Algorithm

def toCSV(data, target):
    with open(target, 'a') as open_csv:
        writer_csv = writer(open_csv)
        writer_csv.writerow(data)       
        open_csv.close()

def plotCSV(target):
    x = []
    y = []
    
    with open(target, 'r') as open_csv:
        plots = reader(open_csv, delimiter=',')
        n = 1
        for row in plots:
            x.append(n)
            y.append(row[0])
            n += 1
        open_csv.close()
        
    plt.figure(2)
    plt.plot(x,y, label='Loaded from file!')
    plt.xlabel('Test')
    plt.ylabel('Mean Squared Error')
    plt.show()

if __name__=="__main__":
    target = 'means.csv'
    #correlationtest()
    toCSV([Regression_Algorithm.runModel()], target)
    #run_PCA_Model() #has a higher R squared value but no corrleation, suggesting hidden variables???
    plotCSV(target)