#import ast
from mediawiki import MediaWiki
import nltk
nltk.download('brown')
from nltk.stem import PorterStemmer  # Stemming interface of NLTK.
from nltk.tokenize import word_tokenize # to perform the work tokenisation
from textblob import TextBlob #Helps to extract the noun phrases of the sentence.

ps = PorterStemmer()

import os
import pandas as pd
import pickle


data_dir = os.path.dirname(os.path.abspath(''))

csv_paths = {'train' : os.path.join(data_dir, "FEVER_data/train.csv") , 'dev' : os.path.join(data_dir, "FEVER_data/shared_task_dev.csv"), 'test' : os.path.join(data_dir, "FEVER_data/shared_task_test.csv") }


wiki = MediaWiki()


def extract_titles(search, docs, stem_claim):
	for p in wiki.search(search, results=2):
		title = p.split('(')[0]
		in_claim = True
		stem_title = [ps.stem(word) for word in word_tokenize(title)]

		for word in stem_title:
			if word not in stem_claim:
				in_claim = False
		if in_claim:
	    		docs.add(p.replace(' ', '_'))   #Since the space is given by the _ in the data we are replacing it similarly to get the rel docs from csv.
	return docs


def find_documents(claim):
	claim_tokenized = word_tokenize(claim)
	claim_stemmed = [ps.stem(token) for token in claim_tokenized]
	
	docs = set()
	for noun_phrase in TextBlob(claim).noun_phrases:
		docs = extract_titles(noun_phrase, docs, claim_stemmed)
	
	return docs


def document_retrieval(task_type, claims=None):  
	if task_type != 'demo':
		csv_data = pd.read_csv(csv_paths[task_type])
		claims = csv_data.claim.values
		verifiable = csv_data.verifiable.values
	#count = 0
	documents = {}
	for i in range(len(claims)):
		
		if task_type != 'train' or verifiable[i] != 'VERIFIABLE':  #The main essence is when we are unable to verify, using it for doc retrieval.
			#count = count + 1
			#print(count)
			claim = claims[i]
			docs = find_documents(claim)
			documents[i] = tuple(docs)

	with open('results/documents_{}.pickle'.format(task_type), 'wb') as f:
		pickle.dump(documents, f)


claims = ['Sancho Panza is a fictional character in a novel written by a Spanish writer born in the 17th century.']
#document_retrieval('train')
#document_retrieval('dev')
#document_retrieval('test')
#with open('results/documents_demo.pickle', 'rb') as f:
#	documents = pickle.load(f)
#	print(documents)
