from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from os import listdir
from os.path import isfile, join
import numpy as np
from numpy.linalg import svd, norm
from numpy import dot

CRANFIELD = 'E:/Teaching/IR/Thuc-hanh/Cranfield'
porter = PorterStemmer()
stops = stopwords.words('english')
def tokenizeFile(fld):
	dic = {}
	file = {}
	i = 0
	for f in listdir(fld):
		fname = join(fld, f)
		if isfile(fname):
			dic[i] = f
			r = open(fname, 'rt', encoding='utf-8')
			terms = {}
			for s in r:
				sents = sent_tokenize(s.strip())
				for sent in sents:
					words = word_tokenize(sent)
					for word in words:
						if not word.isalnum():
							continue
						if word in stops:
							continue
						
						word = porter.stem(word)
						if terms.get(word) == None:
							terms[word] = 1
						else:
							terms[word] += 1
			file[i] = terms
			i += 1			
	return dic, file

def selectTerms(files, n):
	terms = {}
	for ind in files:
		file = files[ind]
		for term in file:
			if terms.get(term) == None:
				terms[term] = file[term]
			else:
				terms[term] += file[term]
	sortterms = sorted(terms.items(), key = lambda item: item[1], reverse=True)
	sortterms = sortterms[:n]
	res = [w for w, i in sortterms]
	return res

def toMatrix(files, terms):
	M = []
	n = len(files)
	for i in range(len(terms)):
		row = [0] * n
		M.append(row)
	for c in files:
		for term in files[c]:
			try:
				r = terms.index(term)
				M[r][c] += files[c][term]
			except:
				continue
	return M

def lsi(files, terms, dim=300):
	A = toMatrix(files, terms)
	S, Sig, Ut = svd(A)
	Z = np.diag(Sig)
	Z = Z[:dim, :dim]
	S = S[:,:dim]
	Ut = Ut[:dim,:]
	K = np.matmul(S, Z)
	D = np.matmul(Z, Ut)
	D = np.transpose(D)
	return K, D

def indexBoolean(fld, filedic):
	ind = {}	
	for i, f in filedic.items():
		fname = join(fld, f)
		if isfile(fname):
			r = open(fname, 'rt', encoding='utf-8')
			for s in r:
				sents = sent_tokenize(s.strip())
				for sent in sents:
					words = word_tokenize(sent)
					for word in words:
						if not word.isalnum():
							continue
						if word in stops:
							continue
						
						word = porter.stem(word)
						if ind.get(word) == None:
							ind[word] = [i]
						else:
							ind[word].append(i)
			r.close()
	return ind

def searchBoolean(ind, qry):
	ret = []
	terms = []
	
	words = word_tokenize(qry)
	for word in words:
		if not word.isalnum():
			continue
		if word in stops:
			continue
		
		word = porter.stem(word)
		terms.append(word)
		if ind.get(word) != None:
			for i in ind[word]:
				if i in ret:
					continue
				ret.append(i)
	
	return ret, terms

def reranking(docs, K, D, terms, qry):
	q = None
	for term in qry:
		if term in terms:
			i = terms.index(term)
			if q is None:
				q = K[i]
			else:
				q = q + K[i]
	
	dlist = {}
	for doc in docs:
		dlist[doc] = dot(D[doc], q)/norm(D[doc])/norm(q)
	
	sortdocs = sorted(dlist.items(), key = lambda item: item[1], reverse=True)	
	res = [w for w, i in sortdocs]
	return res
			
