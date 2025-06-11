import math
import sys
import os
from PyQt5.QtGui import QFont
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

def cilin():
    try:
        cilinPath = os.path.join(current_dir, 'dict_file', 'cilin.txt')
        if not os.path.exists(cilinPath):
            print(f"错误: 同义词库文件不存在: {cilinPath}")
            return {}
            
        f = open(cilinPath, 'r', encoding='utf-8')  # 返回一个文件对象
        line = f.readline()  # 调用文件的 readline()方法
        cilinData = {}
        while line:
            line = f.readline()
            if not line:  # 如果行为空，跳出循环
                break
                
            line = line.strip('\n')
            if len(line) < 9:  # 确保行长度足够
                continue
                
            try:
                code = line[0:8]
                wordsStr = line[9:]
                words = wordsStr.split(' ')
                wordData = []
                for word in words:
                    if len(word) > 1:
                        wordData.append(word)
                if wordData:  # 只添加非空的词列表
                    cilinData[code] = wordData
            except Exception as e:
                print(f"处理同义词库行时出错: {str(e)}, 行内容: {line}")
                continue
                
        f.close()
        if not cilinData:
            print("警告: 同义词库为空")
        return cilinData
    except Exception as e:
        print(f"读取同义词库时出错: {str(e)}")
        return {}


def getSameCode(code1, code2):
    i = 0
    str = ""
    while i < 8 and code1[i] == code2[i]:
        str += code1[i]
        i += 1
    if len(str) == 3 or len(str) == 6:
        str = str[0:-1]
    return str


def getK(code1, code2, length):
    if length == 0 or length == 1 or length == 4:
        k = abs(ord(code1[length]) - ord(code2[length]))
    elif length == 2 or length == 5:
        c1 = code1[length: length + 2]
        c2 = code2[length: length + 2]
        k = abs(int(c1) - int(c2))
    else:
        k = 0
    return k


def getN(sameCode, cilinData):
    if len(sameCode) == 0:
        return 0
    i = 0
    for str in cilinData.keys():
        if str.startswith(sameCode) == 1:
            i += 1
    return i


def simByCilin(word1Code, word2Code, cilinData):
    maxSim = 0
    if len(word1Code) == 0 or len(word2Code) == 0:
        return maxSim
    for code1 in word1Code:
        for code2 in word2Code:
            sameCode = getSameCode(code1, code2)
            length = len(sameCode)
            k = getK(code1, code2, length)
            n = getN(sameCode, cilinData)
            if code1[-1] == '@' or code2[-1] == '@' or length == 0:
                sim = 0.1
            elif length == 1:
                sim = 0.65 * math.cos(n * math.pi / 180) * ((n - k + 1) / n)
            elif length == 2:
                sim = 0.8 * math.cos(n * math.pi / 180) * ((n - k + 1) / n)
            elif length == 4:
                sim = 0.85 * math.cos(n * math.pi / 180) * ((n - k + 1) / n)
            elif length == 5:
                sim = 0.9 * math.cos(n * math.pi / 180) * ((n - k + 1) / n)
            elif length == 8 and sameCode[-1] == '#':
                sim = 0.9
            elif length == 8 and sameCode[-1] == '=':
                sim = 1
            else:
                sim = 0.1
            if sim > maxSim:
                maxSim = sim
    return maxSim


def calculationSim(wordsData):
    cilinData = cilin()
    wordCodeDic = {}  # 记录词语编码的字典
    for word in wordsData:
        codeData = []
        for code, words in cilinData.items():
            if word in words:
                codeData.append(code)
        wordCodeDic[word] = codeData

    # 储存词语语义相关度
    wordsSim = {}
    # 存储词林中缺失的词语
    missingWord = []

    for word1 in wordsData:
        wordSim = {}
        word1Code = wordCodeDic[word1]
        if len(word1Code) == 0:
            missingWord.append(word1)
            continue
        for word2 in wordsData:
            word2Code = wordCodeDic[word2]
            if len(word2Code) == 0:
                missingWord.append(word2)
                continue
            wordSim[word2] = simByCilin(word1Code, word2Code, cilinData)
        wordsSim[word1] = wordSim
        QApplication.processEvents()

    missingWord = list(set(missingWord))

    return wordsSim, missingWord


def getGraph(wordsData, wordsSim):
    minSim = 0.4  # 提高相似度阈值，使边的关系更严格
    graphDatas = {}  # 储存节点间的边
    for word in wordsSim.keys():
        graphData = {}
        for otherWord in wordsSim.keys():
            if word != otherWord:
                sim = wordsSim[word][otherWord]
                if sim > minSim:
                    # 使用指数函数使相似度差异更明显
                    graphData[otherWord] = math.exp(-sim)
        if len(graphData) == 0:
            continue
        graphDatas[word] = graphData
    return graphDatas
