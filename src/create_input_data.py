import xml.dom.minidom

fout = open("../data/Rest_Train.txt","w")

tree = open("../data/Rest_Train_Semeval15.xml", "r")
reviewlines = tree.readlines()

flag = 0
for line in reviewlines:
	#print line
	if '<text>' in line:
		sentence = line.split('text>')[1].split('</text')[0].strip('</')
		#fout.write('\n'+sentence)
		
	if '<Opinion target=' in line:
		temp = line.split('<Opinion target=\"')[1].split('\" category=')
		aspect = temp[0]
		sentiment = temp[1].split('polarity=\"')[1].split('\" from')[0]
		fout.write(sentence+'\t'+aspect+'!~'+sentiment+'\n')		
fout.close()