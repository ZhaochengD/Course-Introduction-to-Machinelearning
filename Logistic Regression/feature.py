#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 01:15:06 2019

@author: zhaocheng_du
"""
import sys

TRAIN_IN = sys.argv[1]
VALID_IN = sys.argv[2]
TEST_IN  = sys.argv[3]
DICT_IN  = sys.argv[4]
TRAIN_OU = sys.argv[5]
VALID_OU = sys.argv[6]
TEST_OU  = sys.argv[7]
FLAG     = int(sys.argv[8])

def convert_dict(filename):
    emp_dict = {}
    f = open(filename)
    for line in f:
        lineLi = line.strip('\n').split(' ')
        emp_dict[lineLi[0]] = lineLi[1]
    return emp_dict

def formatData1(infilename, outfilename, wordDict):
    f = open(infilename)
    o = open(outfilename, 'w')
    for line in f:
        newLine, empDict = line[0], {}
        for word in line[2:].split(' '):
            try:
                idx = wordDict[word]
                empDict.setdefault(idx, '1')
            except:
                continue
        for key, val in empDict.items():
            newLine += '\t' + key + ':' + '1'
        newLine += '\n'
        o.write(newLine)

def formatData2(infilename, outfilename, wordDict):
    f = open(infilename)
    o = open(outfilename, 'w')
    for line in f:
        newLine, empDict = line[0], {}
        for word in line[2:].split(' '):
            try:
                idx = wordDict[word]
                empDict.setdefault(idx, 0)
                empDict[idx] += 1
            except:
                continue
        for key, val in empDict.items():
            if val < 4:
                newLine += '\t' + key + ':' + '1'
        newLine += '\n'
        o.write(newLine)

if __name__ == '__main__':
    wordDict = convert_dict(DICT_IN)
    if FLAG == 1:
        print(11111111111111111)
        formatData1(TRAIN_IN, TRAIN_OU, wordDict)
        formatData1(VALID_IN, VALID_OU, wordDict)
        formatData1(TEST_IN, TEST_OU, wordDict)
    elif FLAG == 2:
        formatData2(TRAIN_IN, TRAIN_OU, wordDict)
        formatData2(VALID_IN, VALID_OU, wordDict)
        formatData2(TEST_IN, TEST_OU, wordDict)  
