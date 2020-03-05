import numpy as np
import matplotlib.pyplot as plt
import sys

TRAIN_INPUT   = sys.argv[1]
INDEX_TO_WORD = sys.argv[2]
INDEX_TO_TAG  = sys.argv[3]
HMMPRIOR      = sys.argv[4]
HMMEMIT       = sys.argv[5]
HMMTRANS      = sys.argv[6]

def readTrain(filename):
    fTrain = open(filename)
    wordLi, labelLi = [], []
    for l in fTrain:
        wordLi_temp, labelLi_temp = [], []
        line = l.strip('\n').split(' ')
        for item in line:
            word, label = item.split('_')
            wordLi_temp.append(word)
            labelLi_temp.append(label)
        wordLi.append(wordLi_temp)
        labelLi.append(labelLi_temp)
    return wordLi, labelLi

def readIndex(filename):
    fIndex = open(filename)
    dIndex = {}
    counter = 0
    for l in fIndex:
        dIndex[l.strip('\n')] = counter
        counter += 1
    return dIndex

class Hmm():
    def __init__(self, wordLi, tagLi, tgIdx, wdIdx):
        self.wordLi = wordLi
        self.tagLi  = tagLi
        self.tgIdx  = tgIdx
        self.wdIdx  = wdIdx
        self.A  = np.zeros((len(self.tgIdx), len(self.tgIdx)))
        self.B  = np.zeros((len(self.tgIdx), len(self.wdIdx)))
        self.Pi = np.zeros((len(self.tgIdx), 1))
        
    def fillPi(self):
        for line in self.tagLi:
            self.Pi[self.tgIdx[line[0]]] += 1
        self.Pi = (self.Pi + 1) / (sum(self.Pi) + len(self.Pi))
    
    def fillA(self):
        for line in self.tagLi:
            for idx in range(len(line) - 1):
                nextIdx = idx + 1
                tagPrev = line[idx]
                tagNext = line[nextIdx]
                self.A[self.tgIdx[tagPrev], self.tgIdx[tagNext]] += 1
        self.A = (self.A + 1) / (np.sum(self.A, axis=1).reshape((-1,1)) + self.A.shape[1])
        
    def fillB(self):
        for lineIdx in range(len(self.tagLi)):
            for tagIdx in range(len(self.tagLi[lineIdx])):
                word = self.wordLi[lineIdx][tagIdx]
                tag  = self.tagLi[lineIdx][tagIdx]
                self.B[self.tgIdx[tag], self.wdIdx[word]] += 1
        self.B = (self.B + 1) / (np.sum(self.B, axis=1).reshape((-1,1)) + self.B.shape[1])
    
    def output(self):
        np.savetxt(HMMPRIOR, self.Pi, delimiter=' ')
        np.savetxt(HMMEMIT, self.B, delimiter=' ')
        np.savetxt(HMMTRANS, self.A, delimiter=' ')

def main():
    wordLi, tagLi = readTrain(TRAIN_INPUT)
    tgIdx = readIndex(INDEX_TO_TAG)
    wdIdx = readIndex(INDEX_TO_WORD)
    hmm = Hmm(wordLi, tagLi, tgIdx, wdIdx)
    hmm.fillPi()
    hmm.fillA()
    hmm.fillB()
    hmm.output()

if __name__ == '__main__':
    main()   