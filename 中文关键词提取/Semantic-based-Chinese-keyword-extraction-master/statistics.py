import jieba.posseg as pseg
from collections import defaultdict
import jieba.analyse as ayse
#ayse.set_idf_path("./idf.txt")

NOT_ALLOW_TAGS = ['x', 'w']
# 词性过滤文件(保留形容词、副形词、名形词、成语、简称略语、习用语、动词、动语素、副动词、名动词、名词、地名、音译地名、机构团体名、其他专名)
ALLOW_SPEECH_TAGS = ['a', 'ad', 'an', 'i', 'j', 'l', 'v', 'vg', 'vd', 'vn', 'n', 'ns', 'nsf', 'nt', 'nz']


def sentence_segmentation(text):
    wordData = []
    psegDataList = pseg.cut(text)
    for data in psegDataList:
        wordData.append(data.word)
    return wordData


def getLoc(wordsData, interDensity, title, firstSentence, lastSentence):
    wordsLoc = defaultdict(float)
    minWord = min(interDensity, key=interDensity.get)
    minValue = interDensity[minWord]
    maxWord = min(interDensity, key=interDensity.get)
    maxValue = interDensity[maxWord]
    # 提取词语的位置特征

    for word in wordsData:
        wordsLoc[word] = 0.5
        if interDensity[word] >= 0.5 * (maxValue + minValue):
            if word in title:
                wordsLoc[word] += 0.5
            elif word in firstSentence or word in lastSentence:
                wordsLoc[word] += 0.3

    return wordsLoc


def getTextRank(length, text):
    try:
        # 尝试使用allowPOS参数
        tags = ayse.textrank(text, topK=length, withWeight=True, allowPOS=ALLOW_SPEECH_TAGS)
    except TypeError:
        # 如果不支持allowPOS参数，则不使用
        tags = ayse.textrank(text, topK=length, withWeight=True)
    textRankScore = defaultdict(float)
    for item in tags:
        textRankScore[item[0]] = item[1]
    return textRankScore


def getTfidf(length, text):
    try:
        # 尝试使用allowPOS参数
        tags = ayse.extract_tags(text, topK=length, withWeight=True, allowPOS=ALLOW_SPEECH_TAGS)
    except TypeError:
        # 如果不支持allowPOS参数，则不使用
        tags = ayse.extract_tags(text, topK=length, withWeight=True)
    tfidf = defaultdict(float)
    for item in tags:
        tfidf[item[0]] = item[1]
    return tfidf


def getFlag(wordsFlag, wordsData):
    flagWeight = defaultdict(float)
    flagWeight['n'] = 1.2  # 提高普通名词权重
    flagWeight['j'] = 0.6  # 降低简称略语权重
    flagWeight['nr'] = 1.5  # 提高人名权重
    flagWeight['ns'] = 1.3  # 提高地名权重
    flagWeight['nsf'] = 1.3  # 提高音译地名权重
    flagWeight['nt'] = 1.4  # 提高机构团体名权重
    flagWeight['nz'] = 1.4  # 提高其他专名权重
    flagWeight['an'] = 0.4  # 降低名形词权重
    flagWeight['l'] = 0.4  # 降低习用语权重
    flagWeight['vn'] = 0.6  # 提高名动词权重
    flagWeight['i'] = 0.3  # 降低成语权重
    flagWeight['a'] = 0.3  # 降低形容词权重
    flagWeight['vd'] = 0.3  # 降低副动词权重
    flagWeight['ad'] = 0.2  # 降低副形词权重
    flagWeight['v'] = 0.2  # 降低动词权重
    flagWeight['vg'] = 0.2  # 降低动语素权重

    wordsFlagWeight = defaultdict(float)
    for word in wordsData:
        wordsFlagWeight[word] = flagWeight[wordsFlag[word]]

    return wordsFlagWeight


def getTextRank1(length, text):
    tags = ayse.textrank(text, topK=length, withWeight=True)
    textRankScore = defaultdict(float)
    for item in tags:
        textRankScore[item[0]] = item[1]
    return textRankScore


def getTfidf1(length, text):
    tags = ayse.extract_tags(text, topK=length, withWeight=True)
    tfidf = defaultdict(float)
    for item in tags:
        tfidf[item[0]] = item[1]
    return tfidf
