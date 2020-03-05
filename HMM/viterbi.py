import numpy as np
import sys
from learnhmm import readTrain, readIndex

TEST_INPUT     = sys.argv[1]
INDEX_TO_WORD  = sys.argv[2]
INDEX_TO_TAG   = sys.argv[3]
HMMPRIOR       = sys.argv[4]
HMMEMIT        = sys.argv[5]
HMMTRANS       = sys.argv[6]
PREDICTED_FILE = sys.argv[7]
METRIC_FILE    = sys.argv[8]

def readModel(PiFilename, AFilename, BFilename):
    Pi = np.genfromtxt(PiFilename, delimiter=' ')
    A = np.genfromtxt(AFilename, delimiter=' ')
    B = np.genfromtxt(BFilename, delimiter=' ')
    return Pi, A, B

class Predictor():
    def __init__(self, Pi, A, B, wordLi, tagLi, tgIdx, wdIdx):
        self.wordLi = wordLi
        self.tagLi  = tagLi
        self.tgIdx  = tgIdx
        self.idxTg  = dict((v,k) for k,v in self.tgIdx.items())
        self.wdIdx  = wdIdx
        self.A  = A
        self.B  = B
        self.Pi = Pi    
    
    def predict(self):
        predLi, staLi = [], []
        for lineIdx in range(len(self.tagLi)):
            wordLine = self.wordLi[lineIdx]
            tagLine  = self.tagLi[lineIdx]
            predLine, staLine = self.predictOneRow(wordLine, tagLine)
            predLi.append(predLine)
            staLi.append(staLine)
        self.predLi = predLi
        self.staLi = staLi
               
    def predictOneRow(self, wordLine, tagLine):
        w  = np.zeros((self.A.shape[0], len(wordLine)))
        p  = np.zeros((self.A.shape[0], len(wordLine)))
        w[:, 0] = np.log(self.Pi * self.B[:, self.wdIdx[wordLine[0]]])
        p[:, 0] = np.array([i for i in range(len(self.Pi))])
        for t in range(1, len(wordLine)):
            for j in range(self.A.shape[0]):
                wTemp = []
                for k in range(self.A.shape[0]):
                    wTemp.append(np.log(self.A[k, j] * self.B[j, self.wdIdx[wordLine[t]]]) + w[k, t-1])
                w[j, t] =  np.max(wTemp)
                p[j, t] =  np.argmax(wTemp)
        pLast = np.argmax(w[:, -1])
        pred_li = []
        for t in range(len(wordLine)-1, -1, -1):
            pred_li.append(pLast)
            pLast = int(p[pLast, t])
        pred_li.reverse()
        return [wordLine[i]+'_'+self.idxTg[pred_li[i]] for i in range(len(pred_li))], [self.idxTg[pred_li[i]] for i in range(len(pred_li))]
    
    def predWrite(self):
        of = open(PREDICTED_FILE, 'w')
        for l in self.predLi:
            for widx in range(len(l)):
                if widx != len(l) - 1:
                    of.write(l[widx] + ' ')
                else:
                    of.write(l[widx])
            of.write('\n')
        of.close()
    
    def metcWrite(self):
        counter, accCounter = 0, 0
        for l in range(len(self.tagLi)):
            for w in range(len(self.tagLi[l])):
                counter += 1
                if self.tagLi[l][w] == self.staLi[l][w]:
                    accCounter += 1
        of = open(METRIC_FILE, 'w')
        self.acc = accCounter / counter
        of.write('Accuracy: {}'.format(accCounter / counter))
        of.close()
        
def main():
    wordLi, tagLi = readTrain(TEST_INPUT)
    tgIdx = readIndex(INDEX_TO_TAG)
    wdIdx = readIndex(INDEX_TO_WORD)
    Pi, A, B = readModel(HMMPRIOR, HMMTRANS, HMMEMIT)
    preder = Predictor(Pi, A, B, wordLi, tagLi, tgIdx, wdIdx)
    preder.predict()
    preder.predWrite()
    preder.metcWrite()
    
if __name__ == '__main__':
    main()
    