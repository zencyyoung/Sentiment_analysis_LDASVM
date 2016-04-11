#!/usr/bin/evn python
#coding:utf-8
import sys
import nltk
import numpy as np
from gensim import corpora,models,matutils
from sklearn.svm import libsvm
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn import svm
from nltk.stem.lancaster import LancasterStemmer

try:
    import xml.etree.cElementTree as ET
    from xml.etree.cElementTree import Element
except ImportError:
    import xml.etree.ElementTree as ET
    from xml.etree.ElementTree import Element

class text_classifier:

    # 初始化
    def __init__(self):
        # 设置过滤词表
        stopwords = self.getStopWords('wordlist_simple.txt')
        english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        self.english_skipwords = []
        self.english_skipwords += english_punctuations
        self.english_skipwords += stopwords
        self.st = LancasterStemmer()
    # 获取
    def getStopWords(self, dir):
        rfile = open(dir)
        words = []
        for line in rfile.readlines():
            for word in line.lower().split():
                words.append(word)
        rfile.close()
        return  words

    def getTokensForCategory(self, sentence):
        sentence = [ word for word in nltk.word_tokenize(sentence.lower())]
        # sentence = nltk.pos_tag(sentence)
        # st = self.st
        tokens = [ word for word in sentence] #if len(word)>3 ]
        filteredTokens = [ self.st.stem(word) for word in tokens if not self.st.stem(word) in self.english_skipwords ]
        return filteredTokens

    def getTokensForPolarity(self,sentence):
        sentence = [ word for word in nltk.word_tokenize(sentence.lower())]
        # sentence = nltk.pos_tag(sentence)
        st = self.st
        tokens = [ word for word in sentence ] #if pos in ['RB','JJ','VBZ']
        filteredTokens = [ st.stem(word) for word in tokens if not st.stem(word) in self.english_skipwords ]
        return filteredTokens

    def create_node(tag, property_map, content):
        '''''新造一个节点
        tag:节点标签
        property_map:属性及属性值map
        content: 节点闭合标签里的文本内容
        return 新节点'''
        element = Element(tag, property_map)
        element.text = content
        return element

    def add_child_node(nodelist, element):
        '''''给一个节点添加子节点
        nodelist: 节点列表
        element: 子节点'''
        for node in nodelist:
            node.append(element)
        return

    def export_xml(self,out_path):
        root = self.tree.getroot()
        num = 0
        for sentence in root.findall('sentence'):
            cat = self.id2cat(self.cat_result[num])
            pol = self.id2pol(self.pol_result[num])
            aspectCategories = Element("aspectCategories")
            category = Element("aspectCategory", {"category":cat,"polarity":pol})
            aspectCategories.append(category)
            sentence.append(aspectCategories)
            num += 1
        self.write_xml(self.tree,out_path)
        print 'XML file exported successfully .'

    def write_xml(self,tree, out_path):
        '''''将xml文件写出
        tree: xml树
        out_path: 写出路径'''
        tree.write(out_path, encoding="utf-8",xml_declaration=True)
        return

    def load_data(self, dir, getTokens):
        # 加载数据
        try:
            self.tree = ET.parse(dir)     #打开xml文档
            #root = ET.fromstring question_string) #从字符串传递xml
            root = self.tree.getroot()         #获得root节点
        except Exception, e:
            print "Error:Cannot parse xml file.", e.message
            sys.exit(1)

        # 初始化,用来存训练数据
        texts = []
        labels = []

        # 遍历xml，并将提取的word,category,sentiment存储到dataDict
        for sentence in root.findall('sentence'): #找到root节点下的所Question节点
            # id = sentence.get('id')
            text = sentence.find('text').text
            words = getTokens(text)
            if sentence.find('aspectCategories') != None:
                for category in sentence.find('aspectCategories'):
                    categoryName = category.get('category')
                    # categories.setdefault(categoryName)
                    polarity = category.get('polarity')
                    texts.append(words)
                    if getTokens == self.getTokensForCategory:
                        labels.append(categoryName)
                    else:
                        labels.append(polarity)
            else:
                texts.append(words)
        return texts, labels

    def cat2id(self,category):
        if category == 'service':
            return 1
        elif category == 'food':
            return 2
        elif category == 'price':
            return 3
        elif category == 'ambience':
            return 4
        else :
            return 5

    def id2cat(self,id):
        if id == 1:
            return 'service'
        elif id == 2:
            return 'food'
        elif id == 3:
            return 'price'
        elif id == 4:
            return 'ambience'
        else :
            return 'anecdotes/miscellaneous'

    def pol2id(self,polarity):
        if polarity == 'positive':
            return 1
        elif polarity == 'negative':
            return 2
        elif polarity == 'neutral':
            return 3
        else :
            return 4

    def id2pol(self,id):
        if id == 1:
            return 'positive'
        elif id == 2:
            return 'negative'
        elif id == 3:
            return 'neutral'
        else :
            return 'conflict'

    # 训练类别分类器
    def load_train_category(self, dir, num_topics = 50):
        self.texts, self.categories = self.load_data(dir,getTokens=self.getTokensForCategory)
        # 生成corpus和tfidf模型
        self.dictionary = corpora.Dictionary(self.texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        # self.tfidf = models.TfidfModel(self.corpus)
        # self.corpus_tfidf = self.tfidf[self.corpus]
        self.lda = models.LdaModel(self.corpus,num_topics = num_topics, id2word = self.dictionary)
        self.corpus_lda = self.lda[self.corpus]
        mat = matutils.corpus2dense(self.corpus_lda, num_terms = num_topics, dtype='float64').T
        lab = []
        for category in self.categories:
            lab.append(self.cat2id(category))
        lab = np.array(lab,dtype='float64')
        self.train(mat,lab)
        return

    # 训练情感分类器
    def load_train_polarity(self, dir):

        self.texts, self.polarities = self.load_data(dir,getTokens=self.getTokensForPolarity)
        # 生成corpus和tfidf模型
        self.dictionary = corpora.Dictionary(self.texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        self.tfidf = models.TfidfModel(self.corpus)
        self.corpus_tfidf = self.tfidf[self.corpus]

        mat = matutils.corpus2dense(self.corpus_tfidf, num_terms=len(self.dictionary),dtype='float64').T
        lab = []
        for polarity in self.polarities:
            lab.append(self.pol2id(polarity))
        lab = np.array(lab,dtype='float64')
        self.train(mat,lab)
        return

    def load_test_category(self, dir):
        texts, categories = self.load_data(dir,getTokens=self.getTokensForCategory)
        # 生成corpus和lda测试集
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        test_cat = self.lda[corpus]
        mat = matutils.corpus2dense(test_cat, num_terms = self.lda.num_topics, dtype='float64').T
        self.cat_result = self.test(mat)
        return self.cat_result

    def load_test_polarity(self, dir):

        texts, polarities = self.load_data(dir,getTokens=self.getTokensForPolarity)
        # 生成corpus和tfidf模型
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        test_pol = self.tfidf[corpus]
        mat = matutils.corpus2dense(test_pol, num_terms=len(self.dictionary),dtype='float64').T
        self.pol_result = self.test(mat)
        return self.pol_result

    def train(self,X,y):
        self.clf = svm.SVC(decision_function_shape='ovr')
	#self.clf = svm.SVC(C=2.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
        self.clf.fit(X,y)

    def test(self,X):
        return self.clf.predict(X)


if __name__ == '__main__':
    ## /Users/marax/PycharmProjects/ML_Text_Classification/ABSA2014/Train/Restaurants_Train.xml
    print 'start'
    tc = text_classifier()
    tc.load_train_category('ABSA2014/Train/Restaurants_Train.xml')
    tc.load_test_category('ABSA2014/Test_PhaseA/Restaurants_Test_PhaseA.xml')
    tc.load_train_polarity('ABSA2014/Train/Restaurants_Train.xml')
    tc.load_test_polarity('ABSA2014/Test_PhaseA/Restaurants_Test_PhaseA.xml')
    tc.export_xml('eval.xml')
    print 'end'
