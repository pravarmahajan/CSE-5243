from bs4 import BeautifulSoup
import time
import os
import numpy as np
import glob
import json
from nltk.corpus import stopwords
import nltk
import preprocessing_config
from nltk.stem.porter import *
import subprocess
import random
from matplotlib import pyplot as plt
import itertools
from collections import defaultdict

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

    regex_tokenizer = nltk.tokenize.RegexpTokenizer(r'[A-Za-z]+')
    stemmer = PorterStemmer()
    article_id = 0
    stops = set(stopwords.words('english'))
    stops.add('reuter')
    lines = []
    num_topics = 0
    for data_element in parsed_documents_with_topics:
        if 'topics' not in data_element.keys() or 'body' not in data_element.keys():
            continue
        if data_element['body'] == '':
            continue
        list_body = set(regex_tokenizer.tokenize(data_element['body']))
        list_body = set([w.lower() for w in list_body])
        topic_labels = data_element['topics']
        
        list_body -= stops
        list_body = map(stemmer.stem, list_body)
        
        topic_labels = [":" + word for word in topic_labels]
        
        antecedents |= set(list_body)
        consequents |= set(topic_labels)
        
        list_body = list_body + topic_labels
        num_topics += len(topic_labels)
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
    f1 = 0
    true_pos = defaultdict(float)
    false_pos = defaultdict(float)
    false_neg = defaultdict(float)
    true_neg = defaultdict(float)

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
        for w in prediction&consequent:
            true_pos[w] += 1
        for w in prediction-consequent:
            false_pos[w] += 1
        for w in consequent-prediction:
            false_neg[w] += 1
        for w in consequent-prediction:
            false_neg[w] += 1
    all_keys = set(true_pos.keys())|set(false_pos.keys())|set(false_neg.keys())
    for key in all_keys:
        if true_pos[key]!=0:
            prec = true_pos[key]/(true_pos[key]+false_pos[key])
            reca = true_pos[key]/(true_pos[key]+false_neg[key])
            f1 += prec*reca/(0.1*(prec+reca))
    f1 /= len(all_keys)
    return f1

def get_sorted_rules():
    with open('data/output/sorted_rules.dat') as f:
        lines = map(str.strip,f.readlines())

    rules = []
    for line in lines:
        consequent, antecedent= map(str.strip, line.split('<-'))
        antecedent = set(antecedent.split())
        rules.append((antecedent, consequent))

    return rules
    
def print_formatted_times(svs, cvs, t1s, t2s, n_rules):
    print "supp\tconf\toff_t\ton_t\tn_rules"
    for (i, (s, c)) in enumerate(itertools.product(svs, cvs)):
        print "%d\t%d\t%0.2f\t%0.2f\t%d" %(s,c,t1s[i],t2s[i], n_rules[i])

def main():
    parsed_docs_file = os.path.join( 
            preprocessing_config.output_data_dir,"parsed_documents.txt")

    print "Loading parsed docs.."
    with open(parsed_docs_file, 'r') as f:
        parsed_docs = json.load(f)
   
    print "creating transactions file"
    convert_to_transaction(parsed_docs)
    
    print "executing apriori algorithm"
    support_values = [2, 5, 10, 20, 40, 70]
    confidence_values = [15, 30, 45, 60, 75, 90]
    #support_values = [2]
    #confidence_values = [75]
    
    values = []
    offline_times = []
    online_times = []
    num_rules = []
    for s in support_values:
        values.append([])
        for c in confidence_values:
            time1 = time.time()
            execute_apriori(s, c)
            sorted_rules = sort_rules()
            num_rules.append(len(sorted_rules))
            time2 = time.time()
            f1 = test()
            time3 = time.time()
            offline_times.append(time2-time1)
            online_times.append(time3-time2)
            values[-1].append(f1)
    values = np.array(values)
    print values
    plt.imshow(values, cmap='hot', interpolation='nearest', origin='lower')
    plt.yticks(range(len(support_values)), support_values)
    plt.xticks(range(len(confidence_values)), confidence_values)
    plt.xlabel("confidence")
    plt.ylabel("support")
    plt.colorbar()
    plt.savefig('plots/heatmap.png')
    print_formatted_times(support_values, confidence_values, offline_times,
                    online_times, num_rules)
    
main()
