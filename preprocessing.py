import parser
import text2vec

def main():
    '''Takes input from sgm files in the data_dir directory, cleans up the xml
    tags and outputs into a human readable json formatted file. Output stored in
    output_data_dir'''
    
    print "Parsing sgm files"
    parsed_documents = parser.parse_xml_files()
    
    '''Construct word frequency matrix given the parsed docuemnts. It saves the
    word frequency matrix to word_freq_matrix.mtx in output_data_dir. labels are
    stored as topics_labels.dat and places_labels.dat'''
    print "Constructing word frequency matrix"
    X_freq = text2vec.construct_word_freq_matrix(parsed_documents)
    
    '''Construct tf-idf matrix given word frequency matrix. The matrix is stored
    in output_data_dir'''
    print "Constructing tf-idf matrix"
    X_tfidf = text2vec.construct_tf_idf_matrix(X_freq)

    text2vec.plot_freq_ranking_graph(X_freq)

if __name__ == "__main__":
    main()
