import xml.dom.minidom,string

fout = open("../data/Rest_train.txt","w")
tree = open("../data/Restaurants_Train_v2_Semeval14.xml", "r")
reviewlines = tree.readlines()

for line in reviewlines:
	#print line
	if '<text>' in line:
		sentence = line.split('text>')[1].split('</text')[0].strip('</')
		exclude = set(string.punctuation) - set(['-','$'])
		sentence = ''.join(ch for ch in sentence.lower() if ch not in exclude)
		
	if '<aspectTerm term=' in line:
		temp = line.split('<aspectTerm term=\"')[1]
		temp1 = temp.split('\" polarity=\"')
		aspect = temp1[0]
		aspect = ''.join(ch for ch in aspect.lower() if ch not in exclude)
		sentiment = temp1[1].split(' from')[0][:-1]
		fout.write(sentence.lower()+'\t'+aspect.lower()+'!~'+sentiment.lower()+'\n')		
fout.close()