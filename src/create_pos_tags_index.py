from collections import defaultdict
import pickle

pos_tag_index = defaultdict()
fin = open('../data/all_pos_tags.txt','r')
count = 0
for line in fin:
	pos_tag_index[line.strip()] = count
	count+=1
pickle.dump(pos_tag_index,open('../data/pos_tag_index.pkl','w'))