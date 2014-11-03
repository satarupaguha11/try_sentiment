import pickle,re,string
from collections import defaultdict

def find_unigrams(sentence):
	unigrams = [word.lower() for word in re.split('\W+',sentence) if word!='']
	return unigrams

def find_bigrams(unigrams):
	bigrams = []
	for i in range(len(unigrams)-1):
		string = unigrams[i]+' '+unigrams[i+1]
		bigrams.append(string)
	return bigrams

def create_vocabulary():
	f = open('../data/Rest_train.txt','r')
	vocab_dict = defaultdict()
	count=0
	for lineno,line in enumerate(f):
		sentence = line.split('\t')[0]
		exclude = set(string.punctuation) - set('-')
		sentence = ''.join(ch for ch in sentence if ch not in exclude)
		unigrams = find_unigrams(sentence)
		bigrams = find_bigrams(unigrams)
		unigrams_bigrams = unigrams+bigrams
		for word in unigrams_bigrams:
			if word.lower() not in vocab_dict:
				vocab_dict[word.lower()]=count
				count+=1
	train_lines = lineno
	f.close()
	f=open('../data/Rest_test.txt','r')
	for lineno,line in enumerate(f):
		sentence = line.split('\t')[0]
		unigrams = find_unigrams(sentence)
		bigrams = find_bigrams(unigrams)
		unigrams_bigrams = unigrams+bigrams
		for word in unigrams_bigrams:
			if word.lower() not in vocab_dict:
				vocab_dict[word.lower()]=count
				count+=1
	test_lines = lineno
	f.close()
	#pickle.dump(vocab_dict,open('../data/vocabulary_dictionary.pkl','w'))
	return train_lines,test_lines,vocab_dict