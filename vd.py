from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import os
import lucene
from org.apache.lucene.store import RAMDirectory
from java.nio.file import Paths, Path
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.analysis.core import StopAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, StringField, TextField
from org.apache.lucene.index import IndexReader
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher, TopDocs
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.analysis.en import EnglishAnalyzer
import random


lucene.initVM()
directory = RAMDirectory()
analyzer = StopAnalyzer()
cf = IndexWriterConfig(analyzer)
cf.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
writer = IndexWriter(directory, cf)
list1 = []

for i in range(1, 256):
    f = open(f"C:\BTTH\CS419\Cranfield\{i}.txt", "r")
    temp = f.readline()
    list1.append(temp)
    f.close()
f1 = open("out.txt", "w")
for i in range(0, 255):
    doc = Document()
    tua = StringField('tua', str(i+1), Field.Store.YES)
    noidung = TextField('noidung', list1[i], Field.Store.NO)
    doc.add(tua)
    doc.add(noidung)
    writer.addDocument(doc)
writer.close()

vectorizer = TfidfVectorizer(stop_words='english')
print("Top terms per cluster:")
a = random.randint(1, 255)
for i in range(0, a):
  X = vectorizer.fit_transform([list1[i]])

  true_k = 1
  model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
  model.fit(X)

  order_centroids = model.cluster_centers_.argsort()[:, ::-1]
  terms = vectorizer.get_feature_names()
  for j in range(true_k):
      print("Cluster %d:" % i),
      f1.write("Cluster %d:\n" % i)
      for ind in order_centroids[j, :10]:
          f1.write(' %s\n' % terms[ind])
          print(' %s' % terms[ind]),
      print
# X = vectorizer.fit_transform(list1)

# true_k = random.randint(1, 255)
# model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
# model.fit(X)

# order_centroids = model.cluster_centers_.argsort()[:, ::-1]
# terms = vectorizer.get_feature_names()
# for j in range(true_k):
#     print("Cluster %d:" % j),
#     f1.write("Cluster %d:\n" % j)
#     for ind in order_centroids[j, :10]:
#         f1.write(' %s\n' % terms[ind])
#         print(' %s' % terms[ind]),
#     print
f1.close()

print("Prediction")

Y = vectorizer.transform(["what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft ."])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["My cat is hungry."])
prediction = model.predict(Y)
print(prediction)