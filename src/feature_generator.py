import re,sys,scipy.io,string,pickle,nltk
from numpy import *
from collections import defaultdict
from create_vocabulary import *

sentiwordnet_lexicon_dictionary = pickle.load(open('../../sentiment_lexicons/sentiwordnet_lexicon_dictionary.pkl','r'))
bingliu_lexicon_dictionary = pickle.load(open('../../sentiment_lexicons/bingliu_lexicon_dictionary.pkl','r'))
mpqa_subj_lexicon_dictionary = pickle.load(open('../../sentiment_lexicons/mpqa_subj_lexicon_dictionary.pkl','r'))
pos_tag_index = pickle.load(open('../data/pos_tag_index.pkl','r'))

def find_context_unigrams(context_size,sentence,aspect):
	#to make sure it is not part of another word. not adding space on both sides to allow the aspect to be the last word in the sentene
	index = sentence.find(' '+aspect)
	if index==-1:
		#not adding space on both sides to allow the aspect to be the first word in the sentene
		index=sentence.find(aspect+' ')
		if index == -1:
			#To accommodate cases where the aspect term is 'it',whereas the sentence contains its
			index=sentence.find(aspect+'s ')
			aspect=aspect+'s'
	left_context=list()
	right_context=list()
	
	if index!=0:
		left_context = sentence[:index]
		#print left_context
	if index<len(sentence)-1:
		right_context = sentence[index+len(aspect)+1:]
		
	if len(left_context)>0:
		unigrams_left = find_unigrams(left_context)
		if len(unigrams_left)>context_size:
			unigrams_left = unigrams_left[-context_size:]
		#print unigrams_left
		if len(right_context)>0:
			unigrams_right = find_unigrams(right_context)
			if len(unigrams_right)>context_size:
				unigrams_right = unigrams_right[:context_size]
			unigrams = unigrams_right+unigrams_left
		else:
			unigrams=unigrams_left

	elif len(right_context)>0:
		unigrams_right = find_unigrams(right_context)
		if len(unigrams_right)>context_size:
			unigrams_right = unigrams_right[:context_size]
		unigrams = unigrams_right
	return unigrams

def find_context_target_bigrams(unigrams,aspect):
	#print unigrams
	context_target_bigrams = list()
	for unigram in unigrams:
		context_target_bigrams.append(unigram+' '+aspect)
	#print context_target_bigrams
	return context_target_bigrams

def surface_features(line,lineno):
	global vocab_dict,feature_matrix

	temp = line.split('\t')
	sentence = temp[0]
	
	exclude = set(string.punctuation) - set('-')
	sentence = ''.join(ch for ch in sentence.lower() if ch not in exclude)
	aspect_senti_pair = temp[1].lower().strip()
	[aspect,sentiment] = aspect_senti_pair.split('!~')
	
	if aspect=='$target$':
		unigrams = find_unigrams(sentence)
	else:
		#context_size = 7
		unigrams = find_context_unigrams(context_size,sentence,aspect)

	#print unigrams
	bigrams = find_bigrams(unigrams)
	context_target_bigrams = find_context_target_bigrams(unigrams,aspect)
	unigrams_bigrams = unigrams+bigrams+context_target_bigrams
	for term in unigrams_bigrams:
		if term in vocab_dict:
			feature_matrix[lineno][vocab_dict[term]] = 1
		if term == aspect:
			feature_matrix[lineno][vocab_dict[term]] = 2
	
	#print sentiment
	if sentiment == 'negative':
		sentiment_label = 1
	elif sentiment == 'positive':
		sentiment_label = 2
	elif sentiment == 'neutral':
		sentiment_label = 3
	else:
		sentiment_label = 4
	return sentiment_label

def lexicon_sentiwordnet(unigrams,lineno):
	no_of_words_in_sentence = len(unigrams)
	num_pos_tokens = 0
	num_neg_tokens = 0
	maximal_sentiment = 0
	sentimentScores = list()
	posSentimentSum = 0
	negSentimentSum = 0
	for unigram in unigrams:
		if unigram in sentiwordnet_lexicon_dictionary:
			posSentimentSum+=float(sentiwordnet_lexicon_dictionary[unigram][0])
			negSentimentSum+=float(sentiwordnet_lexicon_dictionary[unigram][1])
			if float(sentiwordnet_lexicon_dictionary[unigram][0])>float(sentiwordnet_lexicon_dictionary[unigram][1]):
				num_pos_tokens+=1
				sentimentScores.append(float(sentiwordnet_lexicon_dictionary[unigram][0]))
			else:
				num_neg_tokens+=1
				sentimentScores.append(float(sentiwordnet_lexicon_dictionary[unigram][1]))
	if len(sentimentScores)>0:
		maximal_sentiment = max(sentimentScores)
	else:
		maximal_sentiment = 0
	feature_matrix[lineno][len(vocab_dict)]=num_pos_tokens#/float(no_of_words_in_sentence)
	feature_matrix[lineno][len(vocab_dict)+1]=num_neg_tokens#/float(no_of_words_in_sentence)
	feature_matrix[lineno][len(vocab_dict)+2]=maximal_sentiment
	feature_matrix[lineno][len(vocab_dict)+3]=posSentimentSum#/float(no_of_words_in_sentence)
	feature_matrix[lineno][len(vocab_dict)+4]=negSentimentSum#/float(no_of_words_in_sentence)

def lexicon_bingliu(unigrams,lineno):
	no_of_words_in_sentence = len(unigrams)
	num_pos_tokens = 0
	num_neg_tokens = 0
	for unigram in unigrams:
		if unigram in bingliu_lexicon_dictionary['positive']:
			num_pos_tokens+=1
		elif unigram in bingliu_lexicon_dictionary['negative']:
			num_neg_tokens+=1
	feature_matrix[lineno][len(vocab_dict)+5]=num_pos_tokens#/float(no_of_words_in_sentence)
	feature_matrix[lineno][len(vocab_dict)+6]=num_neg_tokens#/float(no_of_words_in_sentence)

def lexicon_mpqa(unigrams,lineno):
	no_of_words_in_sentence = len(unigrams)
	num_pos_tokens = 0
	num_neg_tokens = 0
	for unigram in unigrams:
		if unigram in mpqa_subj_lexicon_dictionary['positive']:
			num_pos_tokens+=1
		elif unigram in mpqa_subj_lexicon_dictionary['negative']:
			num_neg_tokens+=1
	feature_matrix[lineno][len(vocab_dict)+7]=num_pos_tokens#/float(no_of_words_in_sentence)
	feature_matrix[lineno][len(vocab_dict)+8]=num_neg_tokens#/float(no_of_words_in_sentence)

def lexicon_features(line,lineno):
	num_pos_tokens = 0
	num_neg_tokens = 0
	temp = line.split('\t')
	sentence = temp[0]
	aspect = temp[1].split('!~')[0]
	exclude = set(string.punctuation) - set('-')
	sentence = ''.join(ch for ch in sentence.lower() if ch not in exclude)
	if aspect=='$target$':
		unigrams = find_unigrams(sentence)
	else:
		unigrams = find_context_unigrams(context_size,sentence,aspect)
	lexicon_sentiwordnet(unigrams,lineno)
	lexicon_bingliu(unigrams,lineno)
	lexicon_mpqa(unigrams,lineno)

def parse_features_lexical_context(line,lineno):
	temp = line.split('\t')
	sentence = temp[0]
	aspect = temp[1].split('!~')[0]
	unigrams = find_context_unigrams(context_size,sentence,aspect)
	tagged = nltk.pos_tag(unigrams)
	for term in tagged:
		if term[0] in vocab_dict:
			#print term[1]
			feature_matrix[lineno][vocab_dict[term[0]]] = pos_tag_index[term[1]]


def main():
	split = sys.argv[1]
	global vocab_dict,feature_matrix,context_size
	original_data_file = open('../data/train_test_split/Rest_'+split+'.txt','r')
	train_lines,test_lines,vocab_dict = create_vocabulary()
	if split=='train':
		num_sentences=train_lines
	else:
		num_sentences=test_lines
	number_of_lexicon_features = 9
	number_of_parse_features = len(vocab_dict)

	feature_matrix = zeros((num_sentences+1,len(vocab_dict)+number_of_lexicon_features+number_of_parse_features))
	#labels_matrix = zeros((1,1))
	labels_matrix = zeros((num_sentences+1,1))
	context_size = 7
	for lineno,line in enumerate(original_data_file):
		sentiment_label = surface_features(line, lineno)
		labels_matrix[lineno] = sentiment_label
		lexicon_features(line,lineno)
		#parse_features_lexical_context(line,lineno)
		
		#labels_matrix = vstack((labels_matrix, sentiment_label))
	#print labels_matrix,labels_matrix.shape
	
	scipy.io.savemat('../data/features/'+split+'_features.mat', mdict={'data': feature_matrix,'labels':labels_matrix})
	
if __name__=='__main__':
	main()