import pandas as pd
import numpy as np
import jieba
import gensim
import re
from collections import Counter
import jieba.analyse

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib

# 读文件
data1 = pd.read_csv("F:/v20.csv", encoding='utf-8',engine='python',sep=',',names=['会员','级别',	'评价星级',	'评价内容',	'时间',	'点赞数','评论数',	'追评时间',	'追评内容',	'页面网址',	'页面标题',	'采集时间'])#.reset_index()
data1=data1.drop([0],axis=0)

#data1=data1.drop([0],axis=1)

print(len(data1))
content1=list(data1['评价内容'])

xx = ''.join(c for c in content1)

clear = jieba.lcut(xx)
clear= [word for word in clear if len(word) >= 2]
#tfi
keywords = jieba.analyse.extract_tags(xx,topK=20,withWeight=False, allowPOS=())
print(keywords)
#textrank
keywords_textrank = jieba.analyse.textrank(xx,topK=20,withWeight=False,allowPOS=('ns','n'))
print(keywords_textrank)

cleared = pd.DataFrame({'clear': clear})
#print(clear)
stopwords = pd.read_csv("C:/Users/lenovo/Downloads/chineseStopWords.txt", index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')
cleared = cleared[~cleared.clear.isin(stopwords.stopword)]
count_words = cleared.groupby(by=['clear'])['clear'].agg({"num": np.size})
count_words = count_words.reset_index().sort_values(by=["num"], ascending=False)
print(count_words)
# 词云展示
wordcloud = WordCloud(font_path="simhei.ttf", background_color="white", max_font_size=250, width=1300,
                      height=800)  # 指定字体类型、字体大小和字体颜色
word_frequence = {x[0]: x[1] for x in count_words.head(200).values}
wordcloud = wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)
plt.axis("off")
plt.colorbar()  # 颜色条
plt.show()

