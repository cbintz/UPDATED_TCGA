# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 15:02:56 2018
final project machine learning
@author: corinnebintz and lillieatkins
"""
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.feature_extraction import DictVectorizer
import csv
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from numpy import array
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

def main():
    """ Runs program """
    subset = getNarrowedGenes() # subset of 300 genes from narrowed genes dataset with special T-cell expression
    geneData = readGeneFile() # all of our gene data
    peopleData = readPatientFile() # all of our patient data

    survivalLabels = getSurvivalLabels(geneData[1], peopleData[4]) # survival status corresponding to each patient in gene file
    incorrect = verifySurvivalLabels(geneData[1], peopleData[4], survivalLabels)
    print(incorrect)

    # RANDOMLY SELECT 2500 GENES AND CLUSTER WITH DENDROGRAM TO SHOW THAT PATIENTS SEGREGATE BASED ON SURVIVAL STATUS
    # rand = random.sample(range(0, 2500), 2500) # randomly select 2500 genes to use for clustering patients
    # randomGenes = limitGenes(rand, geneData[2]) # get patient gene data for these randomly selected genes
    # randCluster = makeClusters(randomGenes) # make clusters from random genes
    # makeRandomDendrogram(randomGenes[0], geneData[1])

    indices = findGeneIndex(subset, geneData[0]) # get the indices of the gene subset
    geneSubset = limitGenes(indices, geneData[2], geneData[0]) # get the gene data for those indices


    # INDICES OF GENES SELECTED USING FEATURE SELECTION WITH STEP FORWARD SELECTION WITH A RANDOM FOREST CLASSIFIER
    chosenFeatureIndices50 = [4, 5, 6, 11, 13, 14, 21, 25, 26, 27, 28, 32, 36, 41, 44, 48, 51, 70, 71, 77, 78, 81, 89, 93, 98, 100, 107, 112, 113, 115, 119, 123, 127, 130, 131, 138, 140, 141, 143, 145, 148, 156, 158, 163, 169, 183, 184, 185, 187, 200]
    chosenFeaturesIndices30 = [5, 27, 28, 39, 54, 61, 89, 100, 104, 107, 108, 112, 127, 128, 130, 133, 141, 142, 143, 149, 151, 152, 156, 158, 179, 182, 186, 189, 193, 197]


    # PROCESS FOR FEATURE SELECTION USING MUTUAL INFORMATION CLASSIFICATION
    # dv = DictVectorizer()
    # dv.fit(geneSubset[1])
    # selectedFeatures = selectFeaturesMutual(dv, geneSubset[0], a) # mutual info

    # PROCESS FOR FEATURE SELECTION USING STEP FORWARD SELECTION
    # selectedFeatures50 = selectFeatures50(geneSubset[0], a) #sfs
    # print(selectedFeatures)
    # chosenIndices50 = findGeneIndex(selectedFeatures50, geneData[0])
    # print(chosenIndices50)

    # selectedFeatures30 = selectFeatures30(geneSubset[0], a)  # sfs
    # chosenIndices30 = findGeneIndex(selectedFeatures30, geneData[0])
    # print(chosenIndices30)

    chosenGenes = limitGenes(chosenFeatureIndices50, geneData[2], geneData[0]) # get our gene data based on selected genes


    # makeFirstDendrogram(geneSubset[0], geneData[1]) # make dendrogram of gene subset
    # makeNewDendrogram(chosenGenes[0], geneData[1]) # make dendrogram from chosen genes to demonstrate that patients still segregate on survival status

    # newCluster = makeClusters(chosenGenes[0]) # cluster our selected genes

    # survival = findSurvivalRatio(newCluster, geneData[1], peopleData[2], survivalLabels)  # get the survival stats for clusters
    # print(len(survival[0])) # of patients in cluster 1
    # print(survival[2]) # % of patients who survived in cluster 1
    # print(len(survival[1])) # of patients in cluster 2
    # print(survival[3]) # % of patients who survived in cluster 2

    x = chosenGenes[0]

    [training_x, training_y, cvs_x, cvs_y, test_x, test_y] = splitData(x, survivalLabels)
    learning_rate = 0.04
    epochs = 5000

    model = Neural_Network(len(training_x[0]), 50)
    runNN(training_x, training_y, learning_rate, epochs, model)
    testNN(model, test_x, test_y)
    CVSNN(model, cvs_x, cvs_y)


def getSurvivalLabels(genePeople, peopleDict):
    """ Returns survival label corresponding to each patient in gene file"""
    labels = []
    for i in range(len(genePeople)):
        label = peopleDict[genePeople[i]][24]
        labels.append(label)
    return labels

def verifySurvivalLabels(genePeople, peopleDict, newSurvival):
    """ Verify that surival label correctly correspond to each patient in gene file"""
    for i in range(len(genePeople)):
        wrong = 0
        a = peopleDict[genePeople[i]][24]
        b = newSurvival[i]
        if a != b:
            wrong+=1
        return wrong

def runNN(train_x, train_y, alpha, num_epochs, model):
    """ Train our neural net"""
    model.train() #set model to training mode
    print(model.training)  # double check model is in training mode

    criterion = nn.BCELoss(reduction='elementwise_mean') # loss function for binary classification

    optimizer = torch.optim.SGD(model.parameters(), lr=alpha, weight_decay = 0.04) # function to calculate gradient descent

    loss_per_epoch_train = [] # keep track of losses for graph

    epoch_indexes = []

    for epoch in range(num_epochs):

        y_pred = model(train_x) # predictions from model based on training set

        loss = criterion(y_pred, train_y) # calculates loss in epoch

        num_correct = 0
        index = 0

        for prediction in y_pred:
            if prediction < 0.5:
                if train_y[index] == 0: # correclty predicted survival as 0
                    num_correct += 1
            elif prediction >= 0.5:
                if train_y[index] == 1: # correclty predicted survival as 1
                    num_correct += 1
            index += 1

        epoch_acc = num_correct / len(train_x) # accuracy of this epoch

        epoch_loss = loss.item() # gets loss from epoch

        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))

        loss_per_epoch_train.append(epoch_loss) # keep track of loss in this epoch

        optimizer.zero_grad()

        loss.backward() #calculates the gradients

        optimizer.step() #institutes gradient descent

        epoch_indexes.append(epoch)

    # plot our cost curve
    plt.plot(epoch_indexes, loss_per_epoch_train)
    plt.ylabel('J')
    plt.xlabel('Number of Iterations')
    plt.show()

def testNN(model, test_x, test_y):
    """Tests accuracy on test set"""
    model.eval() # set model to evaluation mode
    print(model.training) # double check that model isn't training

    y_pred = model(test_x) # predictions on test set based on our trained model


    num_correct = 0
    index = 0

    for prediction in y_pred:
        if prediction < 0.5:
            if test_y[index] == 0: # correctly predicted survival as 0
                num_correct += 1
        elif prediction >= 0.5:
            if test_y[index] == 1: # correctly predicted survival as 1
                num_correct += 1
            index += 1

    accuracy = num_correct / len(test_y)

    print('Test Acc: {:.4f}'.format(accuracy))

def CVSNN(model, cvs_x, cvs_y):
    """Tests accuracy on cross-validation set"""
    model.eval() # set model to evaluation mode
    print(model.training) # double check that model isn't training

    y_pred = model(cvs_x) # predictions on cross validation set based on our trained model

    num_correct = 0
    index = 0

    for prediction in y_pred:
        if prediction < 0.5:
            if cvs_y[index] == 0:
                num_correct += 1 # correctly predicted survival as 0
        elif prediction >= 0.5:
            if cvs_y[index] == 1: # correctly predicted survival as 1
                num_correct += 1
            index += 1

    accuracy = num_correct / len(cvs_y)


    print('CV Acc: {:.4f}'.format(accuracy))


def limitGenes(geneIndices, patientData, genes):
    """ Limit our gene data to selected genes"""
    newGeneData = []
    newGeneDict = []
    for patient in patientData:
        array = []
        dict = {}
        for i in range(len(patient)):
            if i in geneIndices:
                array.append(patient[i])
                dict[genes[i - 1]] = float(patient[i])
        newGeneData.append(array)
        newGeneDict.append(dict)
    newGeneData = np.array(newGeneData)
    return [newGeneData, newGeneDict]

def findGeneIndex(featureGenes, genes):
    """ Find index of selected genes so we can adjust the data"""
    indices = []
    missing = []
    for gene in featureGenes:
        if gene in genes:
            indices.append(genes.index(gene))
        else:
            missing.append(gene)
    return indices

def findMissingPatients(genePatients, survivalPatients):
    """ Find patients in survival data that are absent in gene data """
    missing = []
    for patient in survivalPatients:
        if patient not in genePatients:
            missing.append(patient)
    return missing


def getNarrowedGenes():
    """ Trim down our data to subset of 300 genes with special T-cell expression"""
    geneIndices = []
    narrowedGenes = []
    with open('narrowedGenes.csv') as geneSubset:
        csv_reader = csv.reader(geneSubset, delimiter=',')
        for row in csv_reader:
            narrowedGenes.append(row[0])
    return narrowedGenes

def selectFeaturesMutual(dv, X, Y):
    """Select top 30 features using mutual information classification: edit indexing to select different # f genes"""
    features = mutual_info_classif(X, Y)
    topFeatures = []
    topFeatureDict ={}
    for score, fname in sorted(zip(features, dv.get_feature_names()), reverse=True)[:30]:
        topFeatures.append(fname)
        topFeatureDict[fname] = score
    return [topFeatures, topFeatureDict]

def selectFeatures50(X, Y):
    """ Select 50 features using step forward selection"""
      #Build RF classifier to use in feature selection
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    # Build step forward feature selection
    sfs1 = sfs(clf,
               k_features=50,
               forward=True,
               floating=False,
               verbose=2,
               scoring='accuracy',
               cv=5)

    # Perform SFS
    sfs1 = sfs1.fit(X, Y)
    feat_cols = list(sfs1.k_feature_idx_)
    print(feat_cols)
    return sfs1

def selectFeatures30(X, Y):
    """ Select 30 features using step forward selection"""
    # Build RF classifier to use in feature selection
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    # Build step forward feature selection
    sfs1 = sfs(clf,
               k_features=30,
               forward=True,
               floating=False,
               verbose=2,
               scoring='accuracy',
               cv=5)

    # Perform SFS
    sfs1 = sfs1.fit(X, Y)
    feat_cols = list(sfs1.k_feature_idx_)
    print(feat_cols)
    return sfs1


def readGeneFile():
    """ Returns array of genes, patients, patientData, and patient: gene sequencing dictionary"""
    patients = []
    patientData = []
    genes = []
    geneDict = []
    with open('transposed.csv') as geneFile:
        line_count = 0
        csv_reader = csv.reader(geneFile, delimiter=',')
        for row in csv_reader:
            if line_count == 0:
                genes.append(row)
                genes = genes[0][1:]
            else:
                intArray = []
                dict = {}
                patients.append(row[0])
                for i in range(1, len(row)):
                    dict[genes[i-1]] = float(row[i])
                    intArray.append(float(row[i]))
                array = np.array(intArray)
                patientData.append(array)
                geneDict.append(dict)
            line_count += 1
    patientData = np.array(patientData)
    return [genes, patients, patientData, geneDict]

def readPatientFile():
    """ Returns array of people, demographics (feature labels), array with each patients data, and patient: feature array dictionary"""
    demographics = []
    people = []
    survivalData = []
    survival = []
    peopleDict = {}
    """ Returns an array of people, demographics, and survivalData"""
    with open('TCGA_LUAD_survival.csv') as survivalFile:
        line_count = 0
        csv_reader = csv.reader(survivalFile, delimiter=',')
        for row in csv_reader:
            if line_count == 0:
                demographics.append(row)
            else:
                array = []
                people.append(row[0])
                survival.append(row[24])
                for i in range(0, len(row)):
                    array.append(row[i])
                peopleDict[row[0]] = array
                array = np.array(array)
                survivalData.append(array)
            line_count += 1
    return [people, demographics, survivalData, survival, peopleDict]

def makeClusters(patientData):
    """ Returns clusters"""
    cluster = AgglomerativeClustering(linkage='ward').fit_predict(patientData)  # perform clustering on patientData and return cluster labels
    return cluster

def makeRandomDendrogram(patientData, patients):
    """ Creates dendrogram from randomly selected genes"""
    linked = linkage(patientData, 'ward')
    plt.figure(figsize=(100, 100))
    dendrogram(linked,  orientation='top',labels=patients,distance_sort='descending',show_leaf_counts=True)
    plt.title("Randomly Selected Gene Dendrogram")
    plt.xlabel("Patients")
    plt.ylabel("Euclidean Distance between points")
    plt.show()

def makeFirstDendrogram(patientData, patients):
    """ Creates initial dendrogram"""
    linked = linkage(patientData, 'ward')
    plt.figure(figsize=(100, 100))
    dendrogram(linked,  orientation='top',labels=patients,distance_sort='descending',show_leaf_counts=True)
    plt.title("Initial Gene Dendrogram")
    plt.xlabel("Patients")
    plt.ylabel("Euclidean Distance between points")
    plt.show()

def makeNewDendrogram(patientData, patients):
    """ Creates new dendrogram for smaller subset of genes"""
    linked = linkage(patientData, 'ward')
    print(linked)
    plt.figure(figsize=(100, 100))
    dendrogram(linked,  orientation='top',labels=patients,distance_sort='descending',show_leaf_counts=True)
    plt.title("Selected Gene Dendrogram")
    plt.xlabel("Patients")
    plt.ylabel("Euclidean Distance between points")
    plt.show()

def findSurvivalRatio(cluster, patients, survivalData, survival):
    """ Returns an array containing each clusters survival rate ratio"""
    #print(survivalData)
    print(cluster)
    cluster0 = []
    survival0 = []
    cluster1 = []
    survival1 = []
    for i in range(len(cluster)):
        if cluster[i] == 0:
            cluster0.append(patients[i])
            survival0.append(survival[i])
            #survival0.append(survivalData[i][23])
        else:
            cluster1.append(patients[i])
            survival1.append(survival[i])
            survival1.append(survivalData[i][23])

    count0 = 0
    count1 = 0
    for i in range(len(survival0)):
        if survival0[i] == "0":
            count0 += 1
    for i in range(len(survival1)):
        if survival1[i] == "0":
            count1 += 1
    cluster0survivalratio = count0/len(survival0)
    cluster1survivalratio = count1/len(survival1)
    return [cluster0, cluster1, cluster0survivalratio, cluster1survivalratio]


def splitData(x_data, y_data):
    """ Split the data into training (80%), cv (10%), and test set (10%)"""
    training_x = []
    training_y = []
    cvs_x = []
    cvs_y = []
    test_x = []
    test_y = []
    for i in range(len(x_data)):
        if i < 412:
            training_x.append(x_data[i])
            training_y.append([float(y_data[i])])
        elif i < 463:
            cvs_x.append(x_data[i])
            cvs_y.append([float(y_data[i])])
        else:
            test_x.append(x_data[i])
            test_y.append([float(y_data[i])])

    #convert to numpy arrays
    training_x = array(training_x, dtype=np.float32)
    training_y = array(training_y, dtype=np.float32)

    cvs_x = array(cvs_x, dtype=np.float32)
    cvs_y = array(cvs_y, dtype=np.float32)
    test_x = array(test_x, dtype=np.float32)
    test_y = array(test_y, dtype=np.float32)

    #convert to tensors and wrap in variables
    training_x = Variable(torch.from_numpy(training_x))
    training_y = Variable(torch.from_numpy(training_y))
    cvs_x = Variable(torch.from_numpy(cvs_x))
    cvs_y = Variable(torch.from_numpy(cvs_y))
    test_x = Variable(torch.from_numpy(test_x))
    test_y = Variable(torch.from_numpy(test_y))

    return [training_x, training_y, cvs_x, cvs_y, test_x, test_y]


class Neural_Network(nn.Module):
    """ Builds our neural net"""
    def __init__(self, input_size, hidden_layers_size):
        super(Neural_Network, self).__init__()

        self.l1 = nn.Linear(input_size, hidden_layers_size) # create input layer
        self.l2 = nn.Linear(hidden_layers_size, hidden_layers_size) # create hidden layer
        self.l3 = nn.Linear(hidden_layers_size, hidden_layers_size) # create hidden layer
        self.l4 = nn.Linear(hidden_layers_size, hidden_layers_size) # create hidden layer
        self.l5 = nn.Linear(hidden_layers_size, 1) # create output layer

        self.sigmoid = torch.nn.Sigmoid() # creates sigmoid function
        self.dropout = nn.Dropout(0.5) # creates dropout function with 50% dropout

    def forward(self, x):
        """ Performs forward propagation through neural network with dropout"""
        out = F.relu(self.l1(x))
        out1 = F.dropout(out)
        out2 = F.relu(self.l2(out1))
        out3 = F.dropout(out2)
        out4 = F.relu(self.l3(out3))
        out5 = F.dropout(out4)
        out6 = F.relu(self.l4(out5))
        out7 = F.dropout(out6)
        y_pred = self.sigmoid(self.l5(out7))
        return y_pred

if __name__ == "__main__":
    main()