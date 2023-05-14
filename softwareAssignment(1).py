import sys
import xml.etree.ElementTree as ET
import string as string
import math as math
import re
from stemming.porter2 import stem as stem
from collections import Counter


class SearchEngine:

    def __init__(self, collectionName, create):
    
        self.collectionName = collectionName

        if create:

            print("Creating index...")

            # Load and parse xml file
            file = collectionName + '.xml'
            tree = ET.parse(file)
            root = tree.getroot()

            # Initialize key and value lists for doc id / doc text dictionary
            key = []
            value = []

            # Loop through all DOC elements in xml file
            for doc in root.iter(tag="DOC"):

                textperdoc = []

                # Get document ID
                doc_id = str(doc.attrib["id"])

                # Append document ID to key list
                key.append(doc_id)

                # Get TEXT element of current DOC element
                text_elem = doc.find('TEXT')

                # Check if text is not broken up by <P> paragraph tags
                if len(text_elem) == 0:

                    # Process text not broken up by <P> tags
                    non_para_text = text_elem.itertext()
                    for word in non_para_text:
                        doctext = word.lower().strip().split()
                        for text in doctext:
                            text = re.sub('[^a-zA-Z]', '', text)  # remove non-alphabetic characters
                            stemmed = stem(text)
                            if stemmed:
                                textperdoc.append(stemmed)

                else:

                    # Process text broken up by <P> tags
                    for i in text_elem:
                        words = i.itertext()
                        for word in words:
                            doctext = word.lower().strip().split()
                            for text in doctext:
                                text = re.sub('[^a-zA-Z]', '', text)  # remove non-alphabetic characters
                                stemmed = stem(text)
                                if stemmed:
                                    textperdoc.append(stemmed)

                # Append document text to value list
                value.append(textperdoc)

            # Create dictionary with document ID as key and text as value
            docdict = dict(zip(key, value))

            # Get total number of documents in corpus
            number_of_docs = len(key)

            # Count unique word occurrences
            values_and_occur_dict = Counter()
            for words in docdict.values():
                for term in set(words):
                    values_and_occur_dict[term] += 1

            # Initialize list of keys and values for idf index dictionary
            idf_index_keys = []
            idf_index_values = []

            # Loop through all words in values_and_occur_dict and calculate idf value for each word
            for key in values_and_occur_dict.keys():
                idf_index_keys.append(key)
                idf = math.log(number_of_docs / values_and_occur_dict[key])
                idf_index_values.append(idf)

            # Create dictionary with word as key and idf value as value
            self.idf_index = dict(zip(idf_index_keys, idf_index_values))

            # Write idf index to file
            with open(collectionName + ".idf", "w") as f:
                for key, value in sorted(self.idf_index.items()):
                    f.write(f"{key}\t{value}\n")

            # Calculating term frequency
            self.occur_dict_by_doc = {}
            for doc_id, doc_text in docdict.items():
                word_counts = {}
                max_count = 0
                for word in sorted(doc_text):
                    count = doc_text.count(word)
                    if count > max_count:
                        max_count = count
                    word_counts[word] = count
                self.tf_scores = dict()
                for word, count in word_counts.items():
                    self.tf_scores[word] = count / max_count
                self.occur_dict_by_doc[doc_id] = self.tf_scores

            # Writing tf scores to file
            with open(collectionName + ".tf", "w") as f:
                for doc_id, tf_scores in self.occur_dict_by_doc.items():
                    for word, tf_score in sorted(tf_scores.items()):
                        f.write(f"{doc_id} {word} {tf_score}\n")

        else:
            print("Reading index from files...")

            # Load idf index
            self.idf_index = {}
            with open(collectionName + '.idf') as f:
                for line in f:
                    try:
                        term, idf = line.strip().split('\t')
                    except ValueError:
                        continue

                    self.idf_index[term] = float(idf)

            # Load tf scores
            self.occur_dict_by_doc = {}
            with open(collectionName + '.tf') as f:
                for line in f:
                    try:
                        doc_id, term, tf = line.strip().split()
                    except ValueError:
                        continue
                    if doc_id not in self.occur_dict_by_doc:
                        self.occur_dict_by_doc[doc_id] = {}
                    self.occur_dict_by_doc[doc_id][term] = float(tf)

    def executeQueryConsole(self):
        '''
        When calling this, the interactive console should be started,
        ask for queries and display the search results, until the user
        simply hits enter.
        '''
        # Start the interactive console and prompt user for query
        query = input("Please enter query, terms separated by whitespace: ")

        # If user hits enter without typing, quit
        if query == "":
            sys.exit(1)

        # Run search on query
        print("Loading...")
        results = self.executeQuery(query)

        # Return results
        if len(results) > 0:
            print("I found the following documents matching your query: ")
        for result in results:
            print(result)

        # Prompt user for another query
        self.executeQueryConsole()

    def executeQuery(self, queryTerms):
        '''
        Input to this function: List of query terms

        Returns the 10 highest ranked documents together with their
        tf.idf-sum scores, sorted score.
        '''

        # Process query text
        query_list = list(queryTerms.lower().strip().split())
        query_stemmed = []
        for word in query_list:
            query_stemmed.append(stem(word))

        # Calculate tf-idf weight for each term in query vector
        query_vector = {}
        for term in query_stemmed:
            if term in self.idf_index:
                query_vector[term] = (1 + math.log(query_stemmed.count(term))) * self.idf_index[term]
        if query_vector == {}:
            if len(query_stemmed) == 1:
                print("Sorry, I didn’t find any documents for this term.")
            else:
                print("Sorry, I didn’t find any documents for those terms.")

        cosine_similarities = {}

        # Iterate over each document and its term frequency vector
        for doc_id, doc_vector in self.occur_dict_by_doc.items():

            # Calculate the numerator of the cosine similarity formula
            numerator = 0
            for term in query_vector.keys():
                # Only calculate if the term appears in both query and document
                if term in doc_vector:  # sus, used to be doc_vector
                    numerator += query_vector[term] * doc_vector[term]

            # Calculate the denominator of the cosine similarity formula
            query_norm = math.sqrt(sum(value ** 2 for value in query_vector.values()))
            doc_norm = math.sqrt(sum(value ** 2 for value in doc_vector.values()))
            denominator = query_norm * doc_norm

            # Calculate the cosine similarity and store it in the dictionary
            if denominator != 0:
                cosine_similarities[doc_id] = numerator / denominator
            else:
                cosine_similarities[doc_id] = 0

        # Rank documents based on cosine similarity scores
        ranked_docs = sorted(cosine_similarities.items(), key=lambda x: x[1], reverse=True)
        out = [(docId, score) for docId, score in ranked_docs[:10] if score > 0]

        return out

if __name__ == '__main__':
   
    # Example to test program
    searchEngine = SearchEngine("nyt199501", create=False)
    searchEngine.executeQueryConsole()
