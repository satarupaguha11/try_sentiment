import sys
split = sys.argv[1]
fin = open('../data/Rest_'+split+'.txt','r')
fout = open('../data/'+split+'_sentences.txt','w')
for line in fin:
	fout.write(line.split('\t')[0]+'\n')