# -*- coding:utf-8 -*-
'''
Created on 20181009
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: XXX
'''
from math import log
import operator

def createDataSet():
    '初始化测试数据集'
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):
    '计算香农熵'
    numEntries = len(dataSet) #数据量
    labelCounts = {} #空字典
    for featVec in dataSet: 
        currentLabel = featVec[-1] #当前标签是最后一个字段值
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0 #如果该标签没在字典中，初始化其值为0
        labelCounts[currentLabel] += 1 #当前标签个数增1
    shannonEnt = 0.0 #初始化香农熵为0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries #当前key概率
        shannonEnt -= prob * log(prob, 2) #当前key的熵
    return shannonEnt #返回熵值

def splitDataSet(dataSet, axis, value):
    retDataSet = [] #初始化列表
    for featVec in dataSet: #逐行取dataSet
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  #截取axis以前的数据
            reducedFeatVec.extend(featVec[axis+1:]) #扩展axis之后的数据，此时reducedFeatVec为去除axis值的featVec
            retDataSet.append(reducedFeatVec) #将上面得到的列表追加到retDataSet
    return retDataSet #返回列表

def chooseBestFeatureToSplit(dataSet):
    '选择最好的分割特征'
    numFeatures = len(dataSet[0]) - 1  #获取特征数
    baseEntropy = calcShannonEnt(dataSet) #基础熵，就是原始数据的熵
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):    #迭代每个特征
        featList = [example[i] for example in dataSet] #将该特征所有值，填入列表
        uniqueVals = set(featList)   #对该特征的所有值，去重
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) #对dataSet数据集按照该特征做切分
            prob = len(subDataSet)/float(len(dataSet)) #特征i，值value 的频率
            newEntropy += prob * calcShannonEnt(subDataSet) #特征i,值value的熵
        infoGain = baseEntropy - newEntropy  #计算熵增，也就是基础熵减少的量
        if (infoGain > bestInfoGain):   
            bestInfoGain = infoGain   
            bestFeature = i  #获取最大熵增时对应的特征i
    return bestFeature     #返回特征的索引

def majorityCnt(classList):
    '获取dataSet中类别名称，该类别的个数是所有类别中最高'
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0 #逐行取dataSet，完成类别个数的初始化以及累增
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #对类别:个数的字典，按照个数逆序排序
    return sortedClassCount[0][0] #取个数最多的类别

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet] #获取dataSet所有类别值组成的列表
    if classList.count(classList[0]) == len(classList): 
        return classList[0] #所有类别相同时，停止分隔，返回该类别名称
    if len(dataSet[0]) == 1: #dataSet中只有一个类别时，停止分隔，返回类别列表中个数最多的类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) #获取最佳特征，返回整型索引值
    bestFeatLabel = labels[bestFeat] #获取最佳特征名称
    myTree = {bestFeatLabel:{}} #初始化数，类型为字典
    del(labels[bestFeat]) #删除标签中该最佳特征
    featValues = [example[bestFeat] for example in dataSet] #获取dataSet中最佳特征所在列的所有数据
    uniqueVals = set(featValues) #将上述数据去重，集合类型
    for value in uniqueVals: #遍历上述集合中每个值
        subLabels = labels[:] #复制所有标签
		#向下生长树，用去除bestFeat特征的剩余数据集继续构建树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree #直到满足递归终止条件，生成完整的树

def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree)[0] #获取树的根节点
    secondDict = inputTree[firstStr] #获取树的第一个键所对应的值，也就是次级字典
    featIndex = featLabels.index(firstStr) #获取firstStr特征所对应的索引
    key = testVec[featIndex] #获取测试向量在该索引所对应的值
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): #如果valueofFeat是字典对象，递归调用分类函数
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat 
    return classLabel #返回类别标签

def storeTree(inputTree, filename):
    '将树写入磁盘文件'
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    '加载该树'
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

