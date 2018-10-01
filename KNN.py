'''
Created on 2018 09 30
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label

@author: xxx
'''
#导入包
import numpy as np
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
    '用已知数据集dataSet&labels预测未知数据集inX分类，投票数量为k'
    dataSetSize = dataSet.shape[0] #取行数
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet #构造和dataSet形状相同的数组，并求两者间差值；即待测点与所有已知点的向量差。
    sqDiffMat = diffMat**2  #上面向量差的平方
    sqDistances = sqDiffMat.sum(axis=1)  #上面平方值，在行的方向的求和，即L2距离的平方
    distances = sqDistances**0.5  #上面结果的开方，即L2 欧氏距离
    sortedDistIndicies = distances.argsort()  #对上面所求距离，顺序排列，返回索引值
    classCount = {} #创建空字典
    for i in range(k):
        '统计出前k个最小距离所覆盖的点，所对应的类别标签数量'
        voteIlabel = labels[sortedDistIndicies[i]] #当前距离点，所对应的标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  #将该标签数量写入字典，注：get函数中的0为初始值
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  #指定key为字典的值，并以此逆向排序；
    return sortedClassCount[0][0]  #第一行第一列，即数量最多的标签

def createDataSet():
    '创建数据组合标签，并返回'
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def file2matrix(filename):
    '文本数据转变成数据矩阵'
    love_dictionary = {'largeDoses':3, 'smallDoses':2, 'didntLike':1}  #声明喜欢程度的数值
    fr = open(filename) #打开文件
    arrayOLines = fr.readlines()  #按行读取
    numberOfLines = len(arrayOLines)  #获取该文件行数
    returnMat = np.zeros((numberOfLines, 3))  #初始化全0矩阵，形状同样本数据
    classLabelVector = []  #初始化列表，用来存储类别标签
    index = 0
    for line in arrayOLines:
        line = line.strip()  #去掉每行数据的两端空格
        listFromLine = line.split('\t')  #分隔每一行数据，即将三个变量返回到一个列表中
        returnMat[index, :] = listFromLine[0:3]  #将上述列表中的值，填充到上文所初始化的全0数组矩阵中
        if(listFromLine[-1].isdigit()):  #判定条件为，上述所得列表最后一个元素，是否为数值型
            classLabelVector.append(int(listFromLine[-1]))  #将上述所得列表最后一个元素转化为整型，追加到上文所初始化的类别标签列表
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))  #不然的话，我们去love_dictionary寻找listFromLine所对应的喜好程度数值标签
        index += 1
    return returnMat, classLabelVector  返回数据矩阵，类别标签列表


def autoNorm(dataSet):
    '该函数为数据标准化处理函数'
    minVals = dataSet.min(0)  #获取每一列最小值
    maxVals = dataSet.max(0)  #获取每一列最大值
    ranges = maxVals - minVals  #求极差
    normDataSet = np.zeros(np.shape(dataSet))  #初始化一个dataSet形状相同的全0数组
    m = dataSet.shape[0]  #获取dataSet行数
    normDataSet = dataSet - np.tile(minVals, (m, 1))  #将dataSet每个元素减去所对应列的最小值
    normDataSet = normDataSet/np.tile(ranges, (m, 1))   #将上述求得的数组中每个元素除以所对应列的极差，即完成标准化处理过程
    return normDataSet, ranges, minVals  #返回标准化数据，极差，最小值

def datingClassTest():
    '用约会网站数据，验证分类模型错误率'
    hoRatio = 0.50  #hold out 10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  #加载数据
    normMat, ranges, minVals = autoNorm(datingDataMat)  #标准化数据
    m = normMat.shape[0]  #获取行数
    numTestVecs = int(m*hoRatio)  #声明测试数据的行数
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)  #用normMat的后50%预测前50%的类别标签
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0  #若预测和实际值不同，累计errorCount，即分类错误的数量
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))  #打印测试数据，错误率
    print(errorCount)

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input(\
                                  "percentage of time spent playing video games?"))  #游戏开始，请输入
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream, ])
    classifierResult = classify0((inArr - \
                                  minVals)/ranges, normMat, datingLabels, 3)  #分类训练
    print("You will probably like this person: %s" % resultList[classifierResult - 1])  #打印结果

def img2vector(filename):
    returnVect = np.zeros((1, 1024))  #初始化1X1024的全0数组，用来存储图片数据
    fr = open(filename)  #打开文件
    for i in range(32): 
        lineStr = fr.readline()  #逐行读取文件
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])  #将32X32的图片数据，铺平填充到上述初始化的1X1024数组中
    return returnVect  #返回转化后的图片数据

def handwritingClassTest():
    hwLabels = []  #初始化手写数据标签
    trainingFileList = listdir('trainingDigits')  #列举文件夹中所有训练数据的文件名，返回到列表
    m = len(trainingFileList)  #获取训练数据的数量
    trainingMat = np.zeros((m, 1024))  #初始化训练数据的数组矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i]  #第i个图片名字
        fileStr = fileNameStr.split('.')[0]  #.分隔，取第一个
        classNumStr = int(fileStr.split('_')[0])  #_分隔，去第一个，即该数据文件对应的数字
        hwLabels.append(classNumStr)  #追加到类别标签列表
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)  #将第i个图片数据填充到训练数据矩阵
    testFileList = listdir('testDigits')  #列举测试数据，将测试图片数据的文件名称返回到列表
    errorCount = 0.0
    mTest = len(testFileList)  #测试图片数据的个数
    for i in range(mTest):
        fileNameStr = testFileList[i]  #以下三行是获取第i个测试图片数据的实际数字
        fileStr = fileNameStr.split('.')[0]  
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)  #加载测试图片数据，格式1X1024
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)  #分类训练
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0  #累计测试数据的错误率
    print("\nthe total number of errors is: %d" % errorCount)  #打印分类错误的数量
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))  #打印测试图片的分类错误率
