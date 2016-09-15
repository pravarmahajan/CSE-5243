from bs4 import BeautifulSoup 
import os
import glob
import json
import pdb

path = "data/"
result = []
for infile in glob.glob(os.path.join(path, "*.sgm")):
	f = open(infile,'r')
	soup = BeautifulSoup(f, 'html.parser')
	soup.prettify()
	
	title = None
	topics = []
	places = []
	body = None
	data = {}
	for reuters in soup.find_all('reuters'):
		title = None
		del topics[:] 
		del places[:] 
		body = None
		data.clear()
		if reuters.title != None:
			title = reuters.title.string
			data["title"] = title
		if reuters.topics != None:
			for topic in reuters.topics.find_all('d'):
				topics.append(topic.string) 
				data["topics"] = list(topics)
		if reuters.places != None:
			for place in reuters.places.find_all('d'):
				places.append(place.string)
				data["places"] = list(places)
		if reuters.body != None:
			body = reuters.body.string.replace('\n', '')
			data["body"] = body
		
		result.append(data.copy())

myfile = open(path + "output.txt", "w")
json.dump(result, myfile, indent=4, separators=(',', ':'))

del result[:]
