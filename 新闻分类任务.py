import pandas as pd
import numpy
import jieba #街吧分词器
df_news = pd.read_table('val.txt', names=['category', 'theme', 'URL', 'content'], encoding='utf-8')
df_news = df_news.dropna()
#print(df_news.head())
#print(df_news.shape)

content = df_news.content.values.tolist()
#print(content[1000])

content_S = []
for line in content:
    current_segment = jieba.lcut(line)
    if len(current_segment) > 1 and current_segment !='\r\n':
        content_S.append(current_segment)

#print(content_S[1000])


df_content = pd.DataFrame({'content_S':content_S})
#print(df_content.head())
stopwords = pd.read_csv("stopwords.txt", index_col=False, sep="\t", quoting=3, names=['stopword'], encoding='utf-8')
#print(stopwords.head())

def drop_stopwords(contents, stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean, all_words

contents = df_content.content_S.values.tolist()
stopwords = stopwords.stopword.values.tolist()
contents_clean, all_words = drop_stopwords(contents, stopwords)

df_content = pd.DataFrame({'contents_clean':contents_clean})
print(df_content.head())

df_all_words = pd.DataFrame({'all_words':all_words})
print(df_all_words.head())
#统计词频
words_count = df_all_words.groupby(by=['all_words'])['all_words'].agg({"count":numpy.size})
words_count = words_count.reset_index().sort_values(by=["count"], ascending=False)
print(words_count.head())

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)

wordcloud = WordCloud(font_path="simhei.ttf", background_color="white", max_font_size=80)
word_frequence = {x[0]:x[1] for x in words_count.head(100).values}
wordcloud = wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)
plt.show()

#提取TF-IDF关键字
import jieba.analyse
index = 1000
print(df_news['content'][index])
content_S_str="".join(content_S[index])
print(" ".join(jieba.analyse.extract_tags(content_S_str, topK=5, withWeight=False)))#topK=5提取关键字次数



#LDA主题模型
from gensim import corpora, models, similarities
import gensim

dictionary = corpora.Dicionary(contents_clean)
corpus = [dictionary.doc2bow(sentence) for sentence in contents_clean]

lda = gensim.models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=20)
print(lda.print_topic(1, topn=5))
for topic in lda.print_top_ics(num_topics=20, num_words=5):
    print(topic[1])