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
    parsed_doc =  os.path.join(
                    preprocessing_config.output_data_dir,
                    "parsed_documents_with_topics_output.dat")
    appearance_file = os.path.join(preprocessing_config.output_data_dir, "appearance.txt")

    regex_tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-z]+')
    stemmer = PorterStemmer()
    article_id = 0
    stops = set(stopwords.words('english'))
    lines = []
    for data_element in parsed_documents_with_topics:
        if 'topics' not in data_element.keys() or 'body' not in data_element.keys():
            continue
        
        list_body = set(regex_tokenizer.tokenize(data_element['body']))
        topic_labels = data_element['topics']
        
        list_body -= stops
        list_body = set(map(stemmer.stem, list_body))
        
        topic_labels = set([":" + word for word in topic_labels])
        
        antecedents |= list_body
        consequents |= topic_labels
        
        list_body |= topic_labels
        lines.append(' '.join(list_body))

        article_id += 1
        
        if article_id % 1000 == 0:
            print "finished working on document # %d" %article_id

    with open(parsed_doc, 'w') as f:
        f.writelines('\n'.join(lines))
    
    with open(appearance_file, 'w') as g:
        g.write('antecedent\n')
        for word in antecedents:
            g.write(word + " antecedent\n")
        
        for label in consequents:
            g.write(label+ " consequent\n")
    
#this function will sort the rules
def sort_rules(rules):
    l = []
    for rule in rules:
        s_c = rule.split("(")[1]
        s_c = s_c.split(",")
        s = float(s_c[0])
        c = float(s_c[1][0:-1])
        l.append((c,s,rule))
        
    rules = sorted(l,key = lambda t:(t[0], t[1]), reverse=True)
    rules = [x[2] for x in rules]
    print(rules)
    
        
def main():
    parsed_docs_file = os.path.join( 
            preprocessing_config.output_data_dir,"parsed_documents.txt")

    print "Loading parsed docs.."
    with open(parsed_docs_file, 'r') as f:
        parsed_docs = json.load(f)

    print "creating transactions file"
    convert_to_transaction(parsed_docs)

    #print "executing apriori algorithm"
    #execute_apriori
    with open(os.path.join( 
            preprocessing_config.output_data_dir,"rules.dat")) as f:
        rules = f.read().splitlines()
        sort_rules(rules)
    
main()
