import xml.dom.minidom,string

fout =  open("../data/temp.txt","w")
tree = open("../data/Restaurants_Train_v2_Semeval14.xml", "r")
reviewlines = tree.readlines()
firstFlag = True
for line in reviewlines:
	
	if '<text>' in line:
		if firstFlag == True:
			firstFlag = False
		else:
			fout.write(sentence.lower())
			for pair in pairs:
				fout.write('\t'+pair[0].lower()+'!~'+pair[1].lower())
			fout.write('\n')		
		sentence = line.split('text>')[1].split('</text')[0].strip('</')
		exclude = set(string.punctuation) - set(['-','$'])
		sentence = ''.join(ch for ch in sentence.lower() if ch not in exclude)
		pairs = list()
		
	if '<aspectTerm term=' in line:
		temp = line.split('<aspectTerm term=\"')[1]
		temp1 = temp.split('\" polarity=\"')
		aspect = temp1[0]
		aspect = ''.join(ch for ch in aspect.lower() if ch not in exclude)
		sentiment = temp1[1].split(' from')[0][:-2]
		pairs.append([aspect,sentiment])

fout.close()

fin =  open("../data/temp.txt","r")
fout = open("../data/Rest_train.txt","w")
for line in fin:
	parts = line.split('\t')
	if len(parts)>1:
		fout.write(line)
fout.close()
