import re,sys,scipy.io,string,pickle,nltk
from numpy import *
from collections import defaultdict
from create_vocabulary import *
from find_parse_context import find_parse_context
sentiwordnet_lexicon_dictionary = pickle.load(open('../../../../sentiment_lexicons/sentiwordnet_lexicon_dictionary.pkl','r'))
bingliu_lexicon_dictionary = pickle.load(open('../../../../sentiment_lexicons/bingliu_lexicon_dictionary.pkl','r'))
mpqa_subj_lexicon_dictionary = pickle.load(open('../../../../sentiment_lexicons/mpqa_subj_lexicon_dictionary.pkl','r'))
nrc_emotion_lexicon_dictionary = pickle.load(open('../../../../sentiment_lexicons/nrc_emotion_lexicon_dictionary.pkl','r'))
sentiment140_lexicon_dictionary = pickle.load(open('../../../../sentiment_lexicons/sentiment140_lexicon_dictionary.pkl','r'))
sentiment_hashtag_lexicon_dictionary = pickle.load(open('../../../../sentiment_lexicons/sentiment_hashtag_lexicon_dictionary.pkl','r'))
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

def surface_features(aspect,sentence,lineno,number_of_pairs):
	previous_features_size = number_of_parse_features+number_of_lexicon_features
	if aspect=='$target$':# or number_of_pairs==1:
		unigrams = find_unigrams(sentence)
	else:
		#context_size = 7
		unigrams = find_context_unigrams(context_size,sentence,aspect)

	#print unigrams
	bigrams = find_bigrams(unigrams)
	context_target_bigrams = find_context_target_bigrams(unigrams,aspect)
	unigrams_bigrams = unigrams+bigrams+context_target_bigrams
	for term in unigrams_bigrams:
		if term in vocab_dict_bigram:		
			feature_matrix[lineno][previous_features_size+vocab_dict_bigram[term]] = 1

def lexicon_sentiwordnet(unigrams,lineno,previous_features_size):
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
	feature_matrix[lineno][previous_features_size]=num_pos_tokens#/float(no_of_words_in_sentence)
	feature_matrix[lineno][previous_features_size+1]=num_neg_tokens#/float(no_of_words_in_sentence)
	feature_matrix[lineno][previous_features_size+2]=maximal_sentiment
	feature_matrix[lineno][previous_features_size+3]=posSentimentSum#/float(no_of_words_in_sentence)
	feature_matrix[lineno][previous_features_size+4]=negSentimentSum#/float(no_of_words_in_sentence)

def lexicon_bingliu(unigrams,lineno,previous_features_size):
	no_of_words_in_sentence = len(unigrams)
	num_pos_tokens = 0
	num_neg_tokens = 0
	for unigram in unigrams:
		if unigram in bingliu_lexicon_dictionary['positive']:
			num_pos_tokens+=1
		elif unigram in bingliu_lexicon_dictionary['negative']:
			num_neg_tokens+=1
	feature_matrix[lineno][previous_features_size+5]=num_pos_tokens
	feature_matrix[lineno][previous_features_size+6]=num_neg_tokens

def lexicon_mpqa(unigrams,lineno,previous_features_size):
	no_of_words_in_sentence = len(unigrams)
	num_pos_tokens = 0
	num_neg_tokens = 0
	for unigram in unigrams:
		if unigram in mpqa_subj_lexicon_dictionary['positive']:
			num_pos_tokens+=1
		elif unigram in mpqa_subj_lexicon_dictionary['negative']:
			num_neg_tokens+=1
	feature_matrix[lineno][previous_features_size+7]=num_pos_tokens#/float(no_of_words_in_sentence)
	feature_matrix[lineno][previous_features_size+8]=num_neg_tokens#/float(no_of_words_in_sentence)

def lexicon_nrc_emotion(unigrams,lineno,previous_features_size):
	no_of_words_in_sentence = len(unigrams)
	num_pos_tokens = 0
	num_neg_tokens = 0
	for unigram in unigrams:
		if unigram in nrc_emotion_lexicon_dictionary.keys():
			if nrc_emotion_lexicon_dictionary[unigram]==1:
				num_pos_tokens+=1
			else:
				num_neg_tokens+=1
	feature_matrix[lineno][previous_features_size+9]=num_pos_tokens#/float(no_of_words_in_sentence)
	feature_matrix[lineno][previous_features_size+10]=num_neg_tokens#/float(no_of_words_in_sentence)

def lexicon_sentiment140(unigrams,lineno,previous_features_size):

	scoreSum = 0
	scores=list()
	for unigram in unigrams:
		if unigram in sentiment140_lexicon_dictionary.keys():
			score = sentiment140_lexicon_dictionary[unigram]
			scoreSum+=score
			scores.append(score)
	maxScore = max(scores)
	feature_matrix[lineno][previous_features_size+11]=scoreSum#/float(no_of_words_in_sentence)
	feature_matrix[lineno][previous_features_size+12]=maxScore#/float(no_of_words_in_sentence)

def lexicon_sentiment_hashtag(unigrams,lineno,previous_features_size):
	scoreSum = 0
	scores=list()
	for unigram in unigrams:
		if unigram in sentiment_hashtag_lexicon_dictionary.keys():
			score = sentiment_hashtag_lexicon_dictionary[unigram]
			scoreSum+=score
			scores.append(score)
	maxScore = max(scores)
	feature_matrix[lineno][previous_features_size+13]=scoreSum#/float(no_of_words_in_sentence)
	feature_matrix[lineno][previous_features_size+14]=maxScore#/float(no_of_words_in_sentence)

def lexicon_features(aspect,sentence,lineno,number_of_pairs):
	
	if aspect=='$target$':# or number_of_pairs==1:
		unigrams = find_unigrams(sentence)
	else:
		unigrams = find_context_unigrams(context_size,sentence,aspect)
	previous_features_size = number_of_parse_features
	lexicon_sentiwordnet(unigrams,lineno,previous_features_size)
	lexicon_bingliu(unigrams,lineno,previous_features_size)
	lexicon_mpqa(unigrams,lineno,previous_features_size)
	lexicon_nrc_emotion(unigrams,lineno,previous_features_size)
	lexicon_sentiment140(unigrams,lineno,previous_features_size)
	lexicon_sentiment_hashtag(unigrams,lineno,previous_features_size)

def parse_features_lexical_context(line,lineno):
	temp = line.split('\t')
	sentence = temp[0]
	aspect = temp[1].split('!~')[0]
	unigrams = find_context_unigrams(context_size,sentence,aspect)
	tagged = nltk.pos_tag(unigrams)
	previous_features_size = number_of_surface_features+number_of_lexicon_features
	for term in tagged:
		if term[0] in vocab_dict_unigram:
			#print term[1]
			feature_matrix[lineno][previous_features_size+vocab_dict_unigram[term[0]]] = pos_tag_index[term[1]]
		'''
		if term[0] == aspect:
			print 'hi'
			feature_matrix[lineno][vocab_dict_unigram[term[0]]] = pos_tag_index[term[1]]+100
		'''
def parse_features_parse_context(aspect,sentence,lineno,sentenceno, number_of_pairs):
	
	if aspect=='$target$':# or number_of_pairs==1:
		unigrams = find_unigrams(sentence)
	else:
		#print sentence
		try:
			#print lineno
			unigrams = find_parse_context(trees_lines[sentenceno],aspect)
		except ValueError:
			unigrams = find_unigrams(sentence)

	previous_features_size = 0#number_of_surface_features+number_of_lexicon_features
	
	for term in unigrams:
		if term in vocab_dict_unigram:
			feature_matrix[lineno][previous_features_size+vocab_dict_unigram[term]] = 1
	'''	
	tagged = nltk.pos_tag(unigrams)
	for term in tagged:
		if term[0] in vocab_dict_unigram:
			#print term[1]
			feature_matrix[lineno][previous_features_size+len(vocab_dict_unigram)+number_of_pos_tags*(vocab_dict_unigram[term[0]]-1)+pos_tag_index[term[1]]] = 1
	'''	
def aspect_features(line,lineno):
	temp = line.split('\t')
	sentence = temp[0]
	aspect = temp[1].split('!~')[0]
	previous_features_size = len(vocab_dict_unigram)+number_of_lexicon_features
	if aspect in vocab_dict_unigram:
		feature_matrix[lineno][previous_features_size+vocab_dict_unigram[aspect]] = 1

def main():
	split = sys.argv[1]
	global vocab_dict_unigram,feature_matrix,context_size,number_of_lexicon_features,trees_lines,number_of_surface_features
	global vocab_dict_bigram,number_of_pos_tags, number_of_parse_features

	original_data_file = open('../data/Rest_'+split+'.txt','r')
	train_lines,number_of_aspects_train,test_lines,number_of_aspects_test,vocab_dict_unigram = create_vocabulary_unigram()
	train_lines,number_of_aspects_train,test_lines,number_of_aspects_test,vocab_dict_bigram = create_vocabulary_bigram()
	number_of_pos_tags = len(pos_tag_index)
	print len(vocab_dict_unigram),len(vocab_dict_bigram)
	
	if split=='train':
		num_sentences=number_of_aspects_train
	else:
		num_sentences=number_of_aspects_test
	print num_sentences
	number_of_lexicon_features = 15
	number_of_surface_features = len(vocab_dict_bigram)
	number_of_parse_features = len(vocab_dict_unigram)# + number_of_pos_tags*len(vocab_dict_unigram)
	trees_file = open('../data/'+split+'_trees_new.txt')
	trees_lines = trees_file.readlines()
	number_of_columns = number_of_lexicon_features+number_of_parse_features+number_of_surface_features
	print number_of_columns
	feature_matrix = zeros((num_sentences,number_of_columns),dtype='int8')
	labels_matrix = zeros((num_sentences,1))
	print labels_matrix.shape,feature_matrix.shape
	context_size = 5
	lineno=-1
	sentenceno=-1
	pos_count=0
	neg_count = 0
	neut_count = 0
	conf_count = 0
	for line in original_data_file:
		sentenceno+=1
		
		temp = line.strip().split('\t')
		sentence=temp[0]
		exclude = set(string.punctuation) - set('-')
		sentence = ''.join(ch for ch in sentence.lower() if ch not in exclude)
		#print sentence
		pairs = temp[1:]
		number_of_pairs = len(pairs)
		#print number_of_pairs
		for i in range(number_of_pairs):
			lineno+=1
			#print lineno
			[aspect, sentiment] = pairs[i].split('!~')
			#print pairs[i]
			parse_features_parse_context(aspect,sentence,lineno,sentenceno,number_of_pairs)
			lexicon_features(aspect,sentence,lineno,number_of_pairs)
			surface_features(aspect,sentence,lineno,number_of_pairs)
			#parse_features_parse_context(aspect,sentence,lineno,sentenceno,number_of_pairs)
			if sentiment == 'negative':
				sentiment_label = 1
				neg_count+=1
			elif sentiment == 'positive':
				sentiment_label = 2
				pos_count+=1
			elif sentiment == 'neutral':
				sentiment_label = 3
				neut_count+=1
			else:
				sentiment_label = 4
				conf_count+=1
			labels_matrix[lineno] = sentiment_label
			#print sentiment_label
	print pos_count,neg_count, neut_count,conf_count
	#print labels_matrix.shape,feature_matrix.shape
	scipy.io.savemat('../data/features/'+split+'_features.mat', mdict={'data': feature_matrix,'labels':labels_matrix})
	
if __name__=='__main__':
	main()