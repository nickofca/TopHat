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
    def __init__(self,docDir,vectors="en_vectors_web_lg",sentLen=15,wordLen=20):
        self.docDir = docDir
        self.nlp = spacy.load(vectors)
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        self.sentLen = sentLen
        self.wordLen = wordLen
        self.tensorList = []
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
                            text = file.read().encode('unicode_escape').decode()
                            sents = list(self.nlp(text).sents)
                        except UnicodeDecodeError:
                            self.UnicodeFail = self.UnicodeFail + 1
                            continue

                        locStart = []
                        locFinish = []
                        for line in annFile:
                            #exclude annotatorNotes
                            if line[0] != "T":
                                continue
                            locStart.append(self.clean(line.split()[2]))
                            locFinish.append(self.clean(line.split()[3]))
                            self.nAnn = self.nAnn +1
                   
                    for sent in sents:
                        sentFlag.add(length=len(sents))
                        #max char length of word plus spacy 300 length vector
                        sentArray = np.zeros((self.sentLen,self.wordLen+303),"int32")
                        labelArray = np.zeros((self.sentLen,4),"int32")
                        
                        for i,word in enumerate(sent):
                            #Get the octal char reading
                            charVec = np.zeros((self.wordLen,),"int32")
                            for j,char in enumerate(word.text):
                                charVec[j] = ord(char)
                                #Prevent overflow of word
                                if j+1 == self.wordLen:
                                    break
                            #Index of word & chars in doc
                            if filename == "12857911.txt" and word.idx>8300:
                                print("ping")
                            index = word.i
                            wstart = word.idx
                            wstop = wstart + len(word)
                            label = 0
                            if wstart in locStart:
                                label = label + 1
                            if wstop in locFinish:
                                label = label + 2
                            #Concat with spacy word vec
                            if word.has_vector: self.nVec = self.nVec + 1
                            sentArray[i,:] = np.concatenate((word.vector,charVec,[index,wstart,wstop]))
                            labelArray[i,label] = 1
                            #Prevent overflow of sentence
                            if i+1 == self.sentLen:
                                break
                        
                        #Join sentArray to tensor
                        self.labelList.append(labelArray)
                        self.tensorList.append(sentArray)
                        
        
        self.tensor = np.array(self.tensorList)
        self.labels = np.array(self.labelList)


class TopFind(object):
    def __init__(self,vectors="en_vectors_web_lg",sent_len=60,width=128,lr=0.001):
        self.nlp = spacy.load(vectors)
        self.sent_len = sent_len
        self.width = width
        self.lr = lr
     
    def preprocess(self):
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        self.embeddings = self.nlp.vocab.vectors.data
     
    def train(self,X,Y,epochs=10,callbacks=None):
        inputs = keras.Input((self.sent_len))
        x = keras.layers.Embedding(self.embeddings.shape[0],self.embeddings.shape[1],
                                       input_length = self.sent_len, trainable = False,
                                       weights = [self.embeddings], mask_zero = True)(inputs)
        x = keras.layers.TimeDistributed(keras.layers.Dense(self.width))(x)
        x = keras.layers.Bidirectional(keras.layers.GRU(self.width))(x)
        x = keras.layers.Dense(4, activation = "sigmoid")(x)
        self.model = keras.Model(inputs = inputs, outputs = x)
        self.model.compile(keras.optimizers.Adam(lr = self.lr),loss = "mse")
        #Train model
        self.model.fit(X,Y,epochs=epochs,callbacks=callbacks)
        self.model.save("models/model.h5")
         
    def predict(self,X,loadFrom=None):
        if self.model is None:
            loadFrom = None
        if loadFrom is not None:
            self.model = keras.models.load_model(loadFrom)
        #Run through preprocess
        self.model.predict(X)
        
 
if __name__ == "__main__":
    trainGen = DocSet("train")
    trainGen.tensorGen()
    trainGen.tensor
         