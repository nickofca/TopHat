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
                        #max char length of word plus spacy 300 length vector
                        vectArray = np.zeros((self.sentLen),"int32")
                        encodeArray = np.zeros((self.sentLen,self.wordLen+2))
                        labelArray = np.zeros((self.sentLen,1),"int32")
                        
                        for i,word in enumerate(sent):
                            #Get the octal char reading
                            charVec = np.zeros((self.wordLen,),"int32")
                            for j,char in enumerate(word.text):
                                charVec[j] = ord(char)
                                #Prevent overflow of word
                                if ord(char) >377:
                                    charVec[j] = 0
                                if j+1 == self.wordLen:
                                    break
                            #Index of word & chars in doc
                            index = word.i
                            wstart = word.idx+1
                            if word.text in topList:
                                labelArray[i,0] = 1
                            #Concat with spacy word vec
                            if word.has_vector: self.nVec = self.nVec + 1
                            vectArray[i] = word.vocab.vectors.find(key=word.orth)
                            if vectArray[i] == -1: vectArray[i] = 0
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
    def __init__(self,vectors="en_vectors_web_lg",sent_len=20,word_len=20,width=32,lr=0.02):
        self.nlp = spacy.load(vectors)
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        self.sent_len = sent_len
        self.word_len = word_len
        self.width = width
        self.lr = lr
        self.embeddings = self.nlp.vocab.vectors.data
     
    def train(self,X1,X2,Y,epochs=25,callbacks=None):
        #embedding for each character in octal (0-377)
        inputsA = keras.Input((self.sent_len,self.word_len))
        xA = keras.layers.Embedding(input_dim=378,output_dim=5)(inputsA)
        xA = keras.layers.TimeDistributed(keras.layers.Bidirectional(keras.layers.GRU(self.width,return_sequences=False,activation = "relu")))(xA)
        modelA = keras.Model(inputs = inputsA, outputs = xA)
        keras.utils.plot_model(modelA, to_file="modelA.png",show_shapes=True)        
        
        #embedding for spaCy vectors
        inputsB = keras.Input((self.sent_len,))
        xB = keras.layers.Embedding(self.embeddings.shape[0],self.embeddings.shape[1],
                           input_length = self.sent_len, trainable = False,
                           weights = [self.embeddings], mask_zero = True)(inputsB)
        xB = keras.layers.TimeDistributed(keras.layers.Dense(self.width,activation = "relu"))(xB)
        modelB = keras.Model(inputs = inputsB, outputs = xB)
        keras.utils.plot_model(modelB, to_file="modelB.png",show_shapes=True)
        
        #inputs for indexing and wstart
        inputsC = keras.Input((self.sent_len,2))
        #xC = keras.layers.TimeDistributed(keras.layers.Dense(self.width,activation = "relu"))(inputsC)
        #modelC = keras.Model(inputs = inputsC, outputs = xC)
        #keras.utils.plot_model(modelC, to_file="modelC.png",show_shapes=True)        
        
        combination = keras.layers.concatenate([modelA.output,modelB.output,inputsC])
        x = keras.layers.Bidirectional(keras.layers.GRU(self.width,return_sequences=True))(xB)
        x = keras.layers.TimeDistributed(keras.layers.Dense(1, activation = "sigmoid",kernel_regularizer = keras.regularizers.l1()))(x)
        self.model = keras.Model(inputs = [modelA.input,modelB.input,inputsC], outputs = x)
        keras.utils.plot_model(self.model, to_file="model.png",show_shapes=True)
        self.model.compile(keras.optimizers.Adam(lr = self.lr),loss = "mse")
        #Train model
        self.model.fit([X1[:,:,:20],X2,X1[:,:,20:]],Y,epochs=epochs,callbacks=callbacks,batch_size=256)
        self.model.save("models/model.h5")
        
        def trainCont(self,X1,X2,Y,epochs=20):
            self.model.fit([X1[:,:,:20],X2,X1[:,:,20:]],Y,epochs=epochs,callbacks=callbacks,batch_size=256)
         
    def predict(self,X1,X2,loadFrom=None):
        if loadFrom is not None:
            self.model = keras.models.load_model(loadFrom)
        #Run through preprocess
        return self.model.predict(X2)
        
def threshGrid(testPred,testGen):
    F1 = 0
    for i in range(0,100):
        thresh = i/100
        print(thresh)
        testPredBinary = testPred>thresh
        print(np.sum(testPredBinary))
        print(F1)
        if np.sum(testPredBinary) == 0:
            break
        precision = np.sum(np.logical_and(testGen.labels,testPredBinary))/np.sum(testPredBinary)
        recall = np.sum(np.logical_and(testGen.labels,testPredBinary))/np.sum(testGen.labels)
        if F1<2*precision*recall/(precision+recall):
            F1 = 2*precision*recall/(precision+recall)
            topThresh = thresh
    return topThresh

if __name__ == "__main__":
    train = True
    predict = True
    
    try:
        trainGen
        print("Data already loaded")
    except NameError:
        trainGen = DocSet("train")
        trainGen.tensorGen()
        testGen = DocSet("test")
        testGen.tensorGen()
    model = TopFind()
    if train:
        model.train(trainGen.encodings,trainGen.vectors,trainGen.labels)
    if predict:
        testPred = model.predict(testGen.encodings,testGen.vectors,loadFrom="models/model.h5")
    thresh = threshGrid(testPred,testGen)
    testPredBinary = testPred>thresh
    rawAcc = sum(sum(testGen.labels==testPredBinary))/(np.prod(testPred.shape))
    precision = np.sum(np.logical_and(testGen.labels,testPredBinary))/np.sum(testPredBinary)
    recall = np.sum(np.logical_and(testGen.labels,testPredBinary))/np.sum(testGen.labels)
    F1 = 2*precision*recall/(precision+recall)

         