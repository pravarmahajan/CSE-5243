from bs4 import BeautifulSoup
import os
import glob
import json
from nltk.corpus import stopwords
import nltk
import preprocessing_config
from nltk.stem.porter import *

# this function takes all .sgm files and it takes the reuter tag whose topic label is not empty, extract body, place, title and topic and store it in json format in parsed_documents_with_topics.txt file. 
def remove_empty_topics():
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
            if type(reuters.title) != type(None):
                title = reuters.title.string
                data["title"] = title
            if type(reuters.topics) != type(None):
                for topic in reuters.topics.find_all('d'):
                    if topic.string != "":
                        topics.append(topic.string)
                        data["topics"] = list(topics)
            if type(reuters.places) != type(None):
                for place in reuters.places.find_all('d'):
                    places.append(place.string)
                    data["places"] = list(places)
            if type(reuters.body) != type(None):
                body = reuters.body.string.replace('\n', ' ')
                data["body"] = body

            if len(topics) != 0:
                result.append(data.copy())

        myfile = open(os.path.join(
                        preprocessing_config.output_data_dir, "parsed_documents_with_topics.txt"),
                        "w"
                    )
        json.dump(result, myfile, indent=4, separators=(',', ':'))
        myfile.close()
        return result

# this function takes parsed_documents_with_topics.txt file, for each reuter tag, it will extract out topic and body, split the body with spaces,remove duplicates, removes stopwords from body, places all keywords of body and topc labels into a list per reuter tag. It writes the list into parsed_documents_with_topics_output.txt file. 
def convert_to_transaction(parsed_documents_with_topics):
    apriori_format_data = []
    list_body = []
    topics_labels = []
    s = set()
    antecedents = set()
    consequents = set()
    
    with open(os.path.join(
                        preprocessing_config.output_data_dir, "parsed_documents_with_topics_output.dat"), 'w') as f, open(os.path.join(preprocessing_config.output_data_dir, "appearance.txt"), 'w') as g:
        article_id = 0
        for data_element in parsed_documents_with_topics:
            try:
                list_body = []
                topics_labels = []
                apriori_format_data = []
                document_string = ""
                if 'body' in data_element.keys():
                    document_string += data_element['body']
                    
                    
                if 'topics' in data_element.keys():
                    topics_labels += data_element['topics']
                
                regex_tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-z]+')
                list_body = regex_tokenizer.tokenize(document_string)
                stemmer = PorterStemmer()
                
                stop_word_filtered_words = [word for word in list_body if word not in stopwords.words('english')]
                stemmed_words = [str(stemmer.stem(word)) for word in stop_word_filtered_words]
                #filtered_words = [word for word in list_body if word not in stopwords.words('english')]
                s = set(stemmed_words)
                list_body = list(s)
                list_body = [word for word in list_body if word not in stopwords.words('english')]
                
                topics_labels = [":" + word for word in topics_labels]
                
                apriori_format_data = list_body + topics_labels
                
                f.write(str(article_id) + " " +  " ".join(apriori_format_data))
                f.write('\n')
                for word in list_body:
                    antecedents.add(word)
                
                for topics_label in topics_labels:
                    consequents.add(topics_label)
                article_id += 1
                
                #documents_as_string.append(document_string)
    
            except Exception as e:
                print(e)
        
        for word in antecedents:
            g.write(word + " antecedent\n")
        
        for label in consequents:
            g.write(label+ " consequent\n")
    

def main():
    with open('data/output/parsed_documents_with_topics.txt') as f:
        parsed_documents_with_topics = json.load(f)
    convert_to_transaction(parsed_documents_with_topics)
    
main()
