import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CustonReview import Reviews

data1 = pd.read_csv("F:/mate20.csv", encoding='utf-8',engine='python',sep=',',names=['会员','级别',	'评价星级',	'评价内容',	'时间',	'点赞数','评论数',	'追评时间',	'追评内容',	'页面网址',	'页面标题',	'采集时间'])#.reset_index()
data=data1.drop([0],axis=0)
color=list(data['评价星级'].dropna().unique())

comments=Reviews(data['评价内容'],data['评价星级'],data['时间'])
print(comments.describe())


# 解决一词多义问题以及统一产品特征名词。比如触摸屏-->触屏等
comments.replace('F:/synonyms.txt')
# 分词。此处用的是结巴分词工具，添加了手机领域的专有词、以及产品特点词语，比如磨砂黑、玫瑰金
comments.segment(product_dict='F:/mobile_dict.txt',stopwords='C:/Users/lenovo/Downloads/chineseStopWords.txt')
# 去除无效评论
initial_words=['经济','杂交','今生今世','红红火火','彰显','荣华富贵','仰慕','滔滔不绝','永不变心','海枯石烂','天崩地裂']
comments.drop_invalid(initial_words=initial_words,max_rate=0.6)
print(comments.describe())

'''
from sklearn import metrics
ss=comments.sentiments(method='snownlp')
ss1=pd.cut(ss,[-0.1,0.0139,0.0315,1],labels=['差评','中评','好评'])
metrics.accuracy_score()
metrics.roc_auc_score()
'''


for k in ['差评','好评','中评']:
    keywords=comments.get_keywords(comments.scores==k)
    print('{} 的关键词为：'.format(k)+'|'.join(keywords))
    print(comments.find_topic(comments.scores == k, n_topics=5))
    filename = 'F:/wordcloud of {}'.format(k)


text_fw=comments.find_keywords('电池|续航|内存|充电|待机')
print(text_fw.head(10))
#kanan 电池的评论情况
print(comments.scores[text_fw.index].value_counts())
features=comments.get_product_features(min_support=0.005)
features_new=list(set(features)-set(['有点','物流','想象','速度快'])\
                  |set(['拍照','照相','内存','续航','全面屏','面容识别','人脸解锁','钥匙串']))
features_opinion,feature_corpus=comments.features_sentiments(features_new,method='score')
features_opinion=features_opinion.sort_values('mention_count',ascending=False)

print(features_opinion[features_opinion['mention_count']>=30]\
      .sort_values('p_positive'))
print('\n好评占比大于所有样本(好评率86.72%)的特征：')
print(features_opinion[(features_opinion['mention_count']>=30)\
                       &(features_opinion['p_positive']>=0.8672)].index)
print('\n好评占比小于所有样本(好评率86.72%)的特征：')
print(features_opinion[(features_opinion['mention_count']>=30)\
                       &(features_opinion['p_positive']<0.8672)].index)

