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

def create_vocabulary_bigram():
	number_of_aspects_train = 0
	f = open('../data/Rest_train.txt','r')
	vocab_dict = defaultdict()
	count=0
	for lineno,line in enumerate(f):
		temp = line.split('\t')
		sentence = temp[0]
		number_of_aspects_train += len(temp[1:])
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
	number_of_aspects_test = 0
	f.close()
	f=open('../data/Rest_test.txt','r')
	for lineno,line in enumerate(f):
		temp = line.split('\t')
		sentence = temp[0]
		number_of_aspects_test += len(temp[1:])
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
	return train_lines,number_of_aspects_train,test_lines,number_of_aspects_test,vocab_dict
	
def create_vocabulary_unigram():
	number_of_aspects_train = 0
	f = open('../data/Rest_train.txt','r')
	vocab_dict = defaultdict()
	count=0
	for lineno,line in enumerate(f):
		temp = line.split('\t')
		sentence = temp[0]
		number_of_aspects_train += len(temp[1:])
		exclude = set(string.punctuation) - set('-')
		sentence = ''.join(ch for ch in sentence if ch not in exclude)
		unigrams = find_unigrams(sentence)
		for word in unigrams:
			if word.lower() not in vocab_dict:
				vocab_dict[word.lower()]=count
				count+=1
	train_lines = lineno
	number_of_aspects_test = 0
	f.close()
	f=open('../data/Rest_test.txt','r')
	for lineno,line in enumerate(f):
		temp = line.split('\t')
		sentence = temp[0]
		sentence = ''.join(ch for ch in sentence if ch not in exclude)
		number_of_aspects_test += len(temp[1:])
		unigrams = find_unigrams(sentence)
		for word in unigrams:
			if word.lower() not in vocab_dict:
				vocab_dict[word.lower()]=count
				count+=1
	test_lines = lineno
	f.close()
	#pickle.dump(vocab_dict,open('../data/vocabulary_dictionary.pkl','w'))
	return train_lines,number_of_aspects_train,test_lines,number_of_aspects_test,vocab_dict
