from bs4 import BeautifulSoup
import os
import glob
import json

path = "data/"
result = []
for infile in glob.glob(os.path.join(path, "*.sgm")):
	f = open(infile,'r')
	soup = BeautifulSoup(f, 'html.parser')
	soup.prettify()

	tit = None
	top = []
	plc = []
	bd = None

	for reuters in soup.find_all('reuters'):
		tit = None
		del top[:] 
		del plc[:] 
		bd = None
		if reuters.title != None:
			tit = reuters.title.string
		if reuters.topics != None:
			for tp in soup.reuters.topics.find_all('d'):
				top.append(tp.string) 
		if reuters.places != None:
			for pl in soup.reuters.places.find_all('d'):
				plc.append(pl.string) 
		if reuters.body != None:
			bd = reuters.body.string
		result.append((tit, top, plc, bd))
		

myfile = open(path+"output.txt", "w")
json.dump(result, myfile, indent=4, separators=(',', ':'))

del result[:]


    	

