from bs4 import BeautifulSoup
import os
import glob
import json
from nltk.corpus import stopwords
import nltk
import preprocessing_config
from nltk.stem.porter import *
import subprocess
import pdb
import random

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
    train_parsed_doc =  os.path.join(
                    preprocessing_config.output_data_dir,
                    "train_parsed_documents_with_topics_output.dat")
    test_parsed_doc =  os.path.join(
                    preprocessing_config.output_data_dir,
                    "test_parsed_documents_with_topics_output.dat")
    appearance_file = os.path.join(preprocessing_config.output_data_dir, "appearance.txt")

    regex_tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-z]+')
    stemmer = PorterStemmer()
    article_id = 0
    stops = set(stopwords.words('english'))
    lines = []
    for data_element in parsed_documents_with_topics:
        if 'topics' not in data_element.keys() or 'body' not in data_element.keys():
            continue
        if data_element['body'] == '':
            continue
        list_body = set(regex_tokenizer.tokenize(data_element['body']))
        topic_labels = data_element['topics']
        
        list_body -= stops
        list_body = map(stemmer.stem, list_body)
        
        topic_labels = [":" + word for word in topic_labels]
        
        antecedents |= set(list_body)
        consequents |= set(topic_labels)
        
        list_body = list_body + topic_labels
        lines.append('%d ' %article_id + ' '.join(list_body))

        article_id += 1
        
        if article_id % 1000 == 0:
            print "finished working on document # %d" %article_id

    random.shuffle(lines)
    split_point = int(0.8*len(lines))
    with open(train_parsed_doc, 'w') as f:
        f.writelines('\n'.join(lines[:split_point]))
    with open(test_parsed_doc, 'w') as f:
        f.writelines('\n'.join(lines[split_point:]))
    
    with open(appearance_file, 'w') as g:
        g.write('antecedent\n')
        for word in antecedents:
            g.write(word + " antecedent\n")
        
        for label in consequents:
            g.write(label+ " consequent\n")
    
#this function will sort the rules
def sort_rules():
    with open('data/output/rules.dat', 'r') as f:
        rules = f.readlines()
    l = []
    for rule in rules:
        rule = rule.strip()
        s_c = rule.split("(")[1]
        s_c = s_c.split(",")
        s = float(s_c[0])
        c = float(s_c[1][:-1])
        l.append((c, s, rule.split("(")[0].strip()))
        
    rules = sorted(l, key=lambda t:(t[0], t[1]), reverse=True)
    rules = [x[2] for x in rules]
    with open(os.path.join(preprocessing_config.output_data_dir,"sorted_rules.dat"), 'w') as f:
        f.writelines('\n'.join(rules))
    return [r.split(' <- ') for r in rules]
    
def execute_apriori(support=10, confidence=30):
    args = ['-tr', '-Rdata/output/appearance.txt',
             '-c%d' %confidence,
             '-s%d' %support,
             'data/output/train_parsed_documents_with_topics_output.dat',
             'data/output/rules.dat'
             ]
    p = subprocess.call(['./apriori' ] + args)

def test():
    rules = get_sorted_rules()
    accuracy = 0
    with open('data/output/test_parsed_documents_with_topics_output.dat') as f:
        lines = map(str.strip, f.readlines())
    for line in lines:
        tokens = set(line.split())
        consequent = set([t for t in tokens if t.startswith(':')])
        antecedent = tokens-consequent
        prediction = set()
        for rule in rules:
            if len(rule[0]-antecedent)==0:
                prediction.add(rule[1])
                
        accuracy += float(len(prediction & consequent))/ \
                    len(prediction | consequent)
    accuracy /= len(lines)
    print "Accuracy = %e" %accuracy

def get_sorted_rules():
    with open('data/output/sorted_rules.dat') as f:
        lines = map(str.strip,f.readlines())

    rules = []
    for line in lines:
        consequent, antecedent= map(str.strip, line.split('<-'))
        antecedent = set(antecedent.split())
        rules.append((antecedent, consequent))

    return rules
    
def main():
    '''parsed_docs_file = os.path.join( 
            preprocessing_config.output_data_dir,"parsed_documents.txt")

    print "Loading parsed docs.."
    with open(parsed_docs_file, 'r') as f:
        parsed_docs = json.load(f)
   
    print "creating transactions file"
    convert_to_transaction(parsed_docs)
    '''
    print "executing apriori algorithm"
    execute_apriori(2, 80)
    
    sorted_rules = sort_rules()
    test()
    
main()
