#!/bin/bash

FILENAME=$1
while read LINE
do
	cd /home/satarupa/Downloads/stanford-corenlp-full-2014-06-16/
	java -cp "*" -mx2g edu.stanford.nlp.parser.lexparser.LexicalizedParser -retainTMPSubcategories -outputFormat "penn" englishPCFG.ser.gz - <<< "$LINE"
done <$FILENAME