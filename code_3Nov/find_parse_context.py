import re,pickle,sys
from collections import defaultdict
from nltk.tree import *
#split = sys.argv[1]
#fin = open('../data/'+split+'_trees_new.txt','r')

def find_parse_context(line,aspect):
	string = line.strip()
	string = string.replace('(','( ')
	string = string.replace(')',' )')
	#aspect = aspect.split(' ')
	#print aspect
	#print string
	#pos_tags_index = ['ROOT','S','NP','JJ','NN','NNP','VP','VBZ','ADJP','RB','PP','IN','PRP$','CC','DT','JJS']
	pos_tags_index = pickle.load(open('../data/pos_tag_index.pkl','r')).keys()
	pos_tags_index += ['NP','PP','ROOT','S','ADJP','VP','SBAR']
	str1 = string.split(' ')
	if ' ' in aspect:

		aspect_terms = aspect.split(' ')
		pos = list()
		
		for term in aspect_terms:
			pos.append(str1.index(term))
		last_index = 0
		
		str1 = str1[:pos[0]]+[aspect]+str1[pos[len(aspect_terms)-1]:]
		
	else:
		if aspect not in str1:
			print 'hi'
			aspect=aspect+'s'
	
	aspect_position = str1.index(aspect)
	#print aspect_position
	distance = defaultdict()
	
	lhs = str1[:aspect_position]
	#print lhs
	for word in range(len(lhs)):
		if lhs[word]!='(' and lhs[word]!=')' and lhs[word] not in pos_tags_index:
			rest = lhs[word:]
			number_of_open_brackets = 0
			
			for word1 in range(len(rest)):
				if rest[word1]=='(':
					number_of_open_brackets+=1
					
				elif rest[word1]==')' and number_of_open_brackets>0:
					number_of_open_brackets-=1
			distance[lhs[word]]=number_of_open_brackets
	selected_words=list()
	rhs = str1[aspect_position:]
	#print rhs
	#print aspect_position
	for word in range(len(rhs)):
		if rhs[word]!='(' and rhs[word]!=')' and rhs[word] not in pos_tags_index:
			rest = rhs[:word]
			#print word
			
			number_of_open_brackets = 0
			
			for word1 in range(len(rest)):
				if rest[word1]=='(':
					number_of_open_brackets+=1
					
				elif rest[word1]==')' and number_of_open_brackets>0:
					number_of_open_brackets-=1
			distance[rhs[word]]=number_of_open_brackets
		
	for dist in distance:
			
		if distance[dist]<=5:
			selected_words.append(dist)
	return selected_words