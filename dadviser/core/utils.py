import os
import re
import time

from nltk import download, data
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as sw

import gensim
import chardet
import logging

from pymorphy2 import MorphAnalyzer
from multiprocessing import Pool, Value, cpu_count

logging.basicConfig(format='%(message)s', level=logging.DEBUG)


class MPValue:
	"""

	"""
	def __init__(self, val=0):
		self.__value = Value('i', val)

	def increment(self, n=1):
		with self.__value.get_lock():
			self.__value.value += n

	@property
	def value(self):
		return self.__value.value

	@value.setter
	def value(self, val):
		with self.__value.get_lock():
			self.__value.value = val


counter = MPValue()
total = MPValue()


def read_file(filepath, encoding=None):
	if encoding is None:
		with open(filepath, mode='rb') as file:
			encoding = chardet.detect(file.read())['encoding']
	with open(filepath, mode='r', encoding=encoding) as file:
		return file.read()


class DataExtractor:
	"""

	"""
	def __init__(self):
		root_path = os.getcwd()
		download_dir = os.path.join(root_path, 'core', 'corpus')

		data.path.append(download_dir)

		download('stopwords', download_dir=download_dir)
		download('punkt', download_dir=download_dir)

		for key in logging.Logger.manager.loggerDict:
			logging.getLogger(key).setLevel(logging.WARNING)

		self.rus = re.compile(r"[а-яё]+")
		self.stopwords_ru = set(sw.words("russian"))
		self.normalize = MorphAnalyzer().normal_forms

	def parallel_file_preparing(self, filename):
		global counter
		# read the file and prepare text for lemmatizing
		text = read_file(filename)
		tokens = self.get_tokens(text)
		# show the process
		# note: += operation is not atomic, so we need to use a lock
		counter.increment()
		if counter.value % 100 == 0:
			logging.debug(f"Finished {counter.value}/{total.value} files")

		return filename, tokens

	def get_tokens(self, text):
		lemmatized_text = ' '.join(self.lemmatize_words(text.lower()))
		return word_tokenize(lemmatized_text, language="russian")

	def lemmatize_words(self, doc):
		"""
		Transforming a words into the normal form (e.g. «машина» instead of «машиной», «на машине», «машинах»)
		"""
		# do normalization of tokens based only on cyrillic words, stopwords excluded
		words = [self.normalize(word)[0] for word in self.rus.findall(doc) if word not in self.stopwords_ru and len(word) > 2]
		return words if len(words) > 2 else None

	def extract(self, filenames):
		global counter, total
		counter.value = 0
		total.value = len(filenames)
		# allocate task on all cores except one
		logging.debug("Parallel tokenizing and lemmatazing processes...")
		start = time.time()
		with Pool(processes=cpu_count() - 1) as pool:
			unordered_file_tokens = pool.map(self.parallel_file_preparing, filenames)
		end = time.time()
		logging.debug(f"Parallelization elapsed {end - start:.2f} seconds")

		return unordered_file_tokens


class DAdviser:
	"""

	"""
	def __init__(self, documents_folder):
		self.data_extractor = DataExtractor()
		self.texts_folder = documents_folder

		self.meta_folder = "data"
		if not os.path.exists(self.meta_folder):
			os.mkdir(self.meta_folder)

		self.corpus_path = os.path.join(self.meta_folder, 'corpus.mm')
		self.tfidf_model_path = os.path.join(self.meta_folder, 'tf_idf.tfidf_model')
		self.dictionary_path = os.path.join(self.meta_folder, 'dictionary.dict')
		self.index_path = os.path.join(self.meta_folder, 'index')

		# if models a calculated before
		if all(map(os.path.exists, (self.corpus_path, self.tfidf_model_path, self.dictionary_path, self.index_path))):
			logging.debug("Loading a corpus, dictionary... Please wait")
			self.corpus = gensim.corpora.MmCorpus(self.corpus_path)
			self.tf_idf = gensim.models.TfidfModel.load(self.tfidf_model_path)
			self.dictionary = gensim.corpora.Dictionary.load(self.dictionary_path)
			features = len(self.dictionary)
			self.sims = gensim.similarities.Similarity(self.index_path, self.tf_idf[self.corpus], features)
			with open(self.index_path) as file:
				self.index_filenames = [filename.strip() for filename in file.readlines()]
		else:   # init them
			self.docs2model()

	def docs2model(self):
		logging.debug("Recoding filenames into the index.txt file")
		abs_path_files = [os.path.join(self.texts_folder, f) for f in os.listdir(self.texts_folder)]
		#
		file_tokens = sorted(self.data_extractor.extract(abs_path_files), key=lambda x: x[0])
		#
		self.index_filenames = [filename for filename, tokens in file_tokens]
		# init filenames
		with open(self.index_path, mode='w', encoding='utf-8') as file:
			file.write("\n".join(self.index_filenames))

		logging.debug("Form the dictionary based on tokens (filenames are ordered)...")
		tokens = [tokens for filename, tokens in file_tokens]
		self.dictionary = gensim.corpora.Dictionary(tokens)
		logging.debug(f"Number of words in dictionary: {len(self.dictionary)}")

		# a bag-of-words representation for a document just lists the number of times each word occurs in the document
		logging.debug("Forming a corpus (a list of bags of words)...")
		self.corpus = list(map(self.dictionary.doc2bow, tokens))

		# tf_idf is a constructor, which calculates inverse document counts for all terms in the training corpus
		logging.debug(f"Forming a tf_idf....")
		self.tf_idf = gensim.models.TfidfModel(self.corpus)
		logging.debug(f"Size of tf_idf: {len(self.tf_idf[self.corpus])}")

		# building the index, and storing index matrix at "index/" folder
		logging.debug("Compute similarities across a collection of documents...")
		self.sims = gensim.similarities.Similarity(self.index_path, self.tf_idf[self.corpus], num_features=len(self.dictionary))

		logging.debug("Save the results...")
		self.dictionary.save(self.dictionary_path)
		self.tf_idf.save(self.tfidf_model_path)
		gensim.corpora.MmCorpus.serialize(self.corpus_path, self.corpus)

		logging.debug("Successfully finished")

	def get_similarity(self, text, toplist=10):
		# tokenize words
		file2_docs = self.data_extractor.get_tokens(text)
		# create bag of words
		query_doc_bow = self.dictionary.doc2bow(file2_docs)
		# find similarity for each document
		query_doc_tf_idf = self.tf_idf[query_doc_bow]
		# sort a files from highest to low similarities (filename, value)
		data = sorted(zip(self.index_filenames, self.sims[query_doc_tf_idf]), key=lambda item: item[1], reverse=True)
		#
		result = []

		toplist = len(data) if toplist > len(data) else toplist

		for top_index in range(toplist):
			filename = data[top_index][0]
			value = data[top_index][1]

			file_index = self.index_filenames.index(filename)
			file_corpus = self.corpus[file_index]
			tf_obj = self.tf_idf[file_corpus]

			top_match = sorted(tf_obj, key=lambda x: x[1], reverse=True)[:10]
			top_terms = [(self.dictionary[word[0]], round(word[1], 4)) for word in top_match]
			result.append(dict(id=top_index + 1, name=filename, percent=f'{value * 100:.2f}', top_terms=top_terms))

		return result