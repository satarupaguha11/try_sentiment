import sys

def edit_trees():
	split = sys.argv[1]
	fin = open('../data/'+split+'_trees.txt','r')
	fout = open('../data/'+split+'_trees_new.txt','w')

	tree = ''
	for line in fin:
		tree+=line
		if line=='\n':
			tree_array = tree.split('\n')
			tree = ''
			for element in tree_array:
				tree+=element
			tree_1 = ' '.join(tree.split())
			fout.write(tree_1+'\n')
			tree = ''
edit_trees()

