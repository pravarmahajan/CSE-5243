from bs4 import BeautifulSoup 
import os
import glob
import json

import preprocessing_config

'''Takes the data directory containing sgm files as input. Returns an array of dictionary containing title, topic, places and body as fields'''
def parse_xml_files():
        result = []
        for infile in glob.glob(os.path.join(preprocessing_config.input_data_dir, "*.sgm")):
                print "Processing %s" %infile
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
                                body = reuters.body.string.replace('\n', ' ')
                                data["body"] = body
                        
                        result.append(data.copy())

        myfile = open(os.path.join(
                        preprocessing_config.output_data_dir, "output.txt"),
                        "w"
                    )
        json.dump(result, myfile, indent=4, separators=(',', ':'))
        myfile.close()
        return result
