#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:01:58 2019

@author: nickofca
"""

import keras
import spacy
import numpy as np
import os
import math

class flag(object):
    def __init__(self,name="Flag"):
        self.name = name
        self.count = 0
     
    def add(self,n=1,verbose=True,length=999):
        self.count = self.count + 1
        if verbose is not False:
            print(f"{self.name}:{self.count}/{length}")
            
    def reset(self):
        self.count = 0
        

class DocSet(object):
    #Consider a better way to incorporate new words to vocab
    def __init__(self,docDir,vectors="en_vectors_web_lg",sentLen=20,wordLen=20):
        self.docDir = docDir
        self.nlp = spacy.load(vectors)
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        self.sentLen = sentLen
        self.wordLen = wordLen
        self.vectList = []
        self.encodeList = []
        self.labelList = []
        self.nAnn = 0
        self.nVec = 0
        self.UnicodeFail = 0
    
    def clean(self,string):
        try:
            return int(string)
        except ValueError:
            if len(string.split(";")):
                return int(string.split(";")[0])
        
    def tensorGen(self):
        fileFlag = flag("File")
        for fileNum,filename in enumerate(os.listdir(self.docDir)):
            fileFlag.add(length=len(os.listdir(self.docDir)))
            if filename.endswith(".txt"): 
                with open(os.path.join(self.docDir,filename),"r") as file:
                    #Create annotations tags
                    #Reload data
                    with open(os.path.join(self.docDir,filename)[:-3]+"ann","r", encoding="ISO-8859-1") as annFile:
                        #Be sure that embeddings seperate by sentence
                        #For some reason it returns document wide
                        sentFlag = flag("  Sent")
                        print(filename)
                        try:
                            #get the raw literas strings
                            text = file.read()
                            sents = list(self.nlp(text).sents)
                        except UnicodeDecodeError:
                            self.UnicodeFail = self.UnicodeFail + 1
                            continue

                        topList = []
                        for line in annFile:
                            #exclude annotatorNotes
                            if line[0] != "T":
                                continue
                            topList.append(line.split()[4])
                            self.nAnn = self.nAnn +1
                   
                    for sent in sents:
                        sentFlag.add(length=len(sents))
                        #max char length of word plus spacy 300 length vector
                        vectArray = np.zeros((self.sentLen,300),"int32")
                        encodeArray = np.zeros((self.sentLen,self.wordLen+2))
                        labelArray = np.zeros((self.sentLen,1),"int32")
                        
                        for i,word in enumerate(sent):
                            #Get the octal char reading
                            charVec = np.zeros((self.wordLen,),"int32")
                            for j,char in enumerate(word.text):
                                charVec[j] = ord(char)
                                #Prevent overflow of word
                                if j+1 == self.wordLen:
                                    break
                            #Index of word & chars in doc
                            if filename == "2780295.txt" :
                                print("ping")
                            index = word.i
                            wstart = word.idx+1
                            if word.text in topList:
                                labelArray[i,0] = 1
                            #Concat with spacy word vec
                            if word.has_vector: self.nVec = self.nVec + 1
                            vectArray[i,:] = word.vector
                            encodeArray[i,:] = np.concatenate((charVec,[index,wstart]))
                            #Prevent overflow of sentence
                            if i+1 == self.sentLen:
                                break
                        
                        #Join sentArray to tensor
                        self.labelList.append(labelArray)
                        self.vectList.append(vectArray)
                        self.encodeList.append(encodeArray)
                        
        
        self.encodings = np.array(self.encodeList)
        self.vectors = np.array(self.vectList)
        self.labels = np.array(self.labelList)


class TopFind(object):
    def __init__(self,vectors="en_vectors_web_lg",sent_len=20,width=128,lr=0.001):
        self.nlp = spacy.load(vectors)
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        self.sent_len = sent_len
        self.width = width
        self.lr = lr
        self.embeddings = self.nlp.vocab.vectors.data
     
    def train(self,X1,X2,Y,epochs=10,callbacks=None):
        inputsA = keras.Input((self.sent_len,22))
        xA = keras.layers.TimeDistributed(keras.layers.Dense(self.width))(inputsA)
        modelA = keras.Model(inputs = inputsA, outputs = xA)
        keras.utils.plot_model(modelA, to_file="modelA.png",show_shapes=True)        
        
        inputsB = keras.Input((self.sent_len,300))
        xB = keras.layers.TimeDistributed(keras.layers.Dense(self.width))(inputsB)
        modelB = keras.Model(inputs = inputsB, outputs = xB)
        keras.utils.plot_model(modelB, to_file="modelB.png",show_shapes=True)
        
        combination = keras.layers.concatenate([modelA.output,modelB.output])
        x = keras.layers.Bidirectional(keras.layers.GRU(self.width,return_sequences=True))(combination)
        x = keras.layers.TimeDistributed(keras.layers.Dense(1, activation = "sigmoid"))(x)
        self.model = keras.Model(inputs = [modelA.input,modelB.input], outputs = x)
        keras.utils.plot_model(self.model, to_file="model.png",show_shapes=True)
        self.model.compile(keras.optimizers.Adam(lr = self.lr),loss = "mse")
        #Train model
        self.model.fit([X1,X2],Y,epochs=epochs,callbacks=callbacks)
        self.model.save("models/model.h5")
         
    def predict(self,X1,X2,loadFrom=None):
        if self.model is None:
            loadFrom = None
        if loadFrom is not None:
            self.model = keras.models.load_model(loadFrom)
        #Run through preprocess
        return self.model.predict([X1,X2])
        
def threshGrid(testPred,testGen):
    F1 = 0
    for i in range(0,100):
        thresh = i/100
        print(thresh)
        testPredBinary = testPred>thresh
        rawAcc = np.sum(testGen.labels==testPredBinary)/(np.prod(testPred.shape))
        precision = np.sum(np.logical_and(testGen.labels,testPredBinary))/np.sum(testPredBinary)
        recall = np.sum(np.logical_and(testGen.labels,testPredBinary))/np.sum(testGen.labels)
        if F1<2*precision*recall/(precision+recall):
            F1 = 2*precision*recall/(precision+recall)
    return F1

if __name__ == "__main__":
    train = False
    predict = False
    
    try:
        trainGen
        print("Data already loaded")
    except NameError:
        trainGen = DocSet("train")
        trainGen.tensorGen()
        testGen = DocSet("test")
        testGen.tensorGen()
    if train:
        model = TopFind()
    if predict:
        model.train(trainGen.encodings,trainGen.vectors,trainGen.labels)
        testPred = model.predict(testGen.encodings,testGen.vectors)
    thresh = threshGrid(testPred,testGen)
    testPredBinary = testPred>thresh
    rawAcc = sum(sum(testGen.labels==testPredBinary))/(np.prod(testPred.shape))
    precision = np.sum(np.logical_and(testGen.labels,testPredBinary))/np.sum(testPredBinary)
    recall = np.sum(np.logical_and(testGen.labels,testPredBinary))/np.sum(testGen.labels)
    F1 = 2*precision*recall/(precision+recall)

         