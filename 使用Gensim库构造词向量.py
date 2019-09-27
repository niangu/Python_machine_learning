from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
raw_sentences = ["the quick brown fox jumps over the lazy dogs", "yoyoyo you go home now to sleep"]

sentences = [s.split() for s in raw_sentences]
print(sentences)

model = word2vec.Word2Vec(sentences, min_count=1)
print(model.similarity('dogs', 'you'))#对比相似度
#min_count：在不同大小的语料集中，我们对于基准词频的需求是不一样的。比如在较大的语料集中，我们希望忽略那些只出现过一俩次的的单词，
#这里我们就可以通过设置min_count参数进行控制。一般而言，合理的参数值会设置在0-100之间
#Size:size参数主要是用来设置神经网络的层数，Word2Vec中的默认值是设置为100层。更大的层次设置意味着更多的输入数据，不过也能提升整体的准确度，
#合理的设置范围为10～数百
