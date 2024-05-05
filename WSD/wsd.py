from nltk.corpus import wordnet # This might require "nltk.download('wordnet')" and "nltk.download('omw-1.4')".
import math
import random
from nltk.tokenize import word_tokenize
from utils import *
from collections import Counter, defaultdict

class WSDClassifier(object):
    """
    Abstract class for WSD classifiers
    """

    def evaluate(self, instances):
        """
        Evaluates the classifier on a set of instances.
        Returns the accuracy of the classifier, i.e. the percentage of correct predictions.
        
        instances: list[WSDInstance]
        """
        
        accuracy = sum([1 if self.predict_sense(instance) == instance.sense else 0 for instance in instances]) / len(instances)
        return accuracy * 100 # return rounded percentage

class RandomSense(WSDClassifier):
    """
    RandomSense baseline
    """
    
    def __init__(self):
        pass # Nothing to do.

    def train(self, instances=[]):
        """
        instances: list[WSDInstance]
        """
        
        pass # Nothing to do.

    def predict_sense(self, instance):
        """
        instance: WSDInstance
        """
        
        senses = list(WN_CORRESPONDANCES[instance.lemma].keys()) # list[string]
        random.shuffle(senses)
        return senses[0]
    
    def __str__(self):
        return "RandomSense"

class MostFrequentSense(WSDClassifier):
    """
    Most Frequent Sense baseline
    """
    
    def __init__(self):
        self.mfs = None # Should be defined as a dictionary from lemmas to most frequent senses (dict[string -> string]) at training.
    
    def train(self, instances):
        """
        instances: list[WSDInstance]
        """

        # initialize a dictionary from lemmas (strings) to list of senses
        # (eventually the most common one in this list will be picked)
        self.mfs = {instance.lemma: [] for instance in instances}
        # loop over all instances and append its sense to a lemma
        for instance in instances:
            self.mfs[instance.lemma].append(instance.sense)
        for lemma in self.mfs:
            self.mfs[lemma] = Counter(self.mfs[lemma]).most_common()[0][0] # pick the most common sense for mapping

        return self.mfs # dictionary (str->str)


    def predict_sense(self, instance):
        """
        instance: WSDInstance
        """

        return self.mfs[instance.lemma] # returns most common sense in data

    def __str__(self):
        return "MostFrequentSense"

class SimplifiedLesk(WSDClassifier):
    """
    Simplified Lesk algorithm
    """
    
    def __init__(self):
        """
        """
        
        self.signatures = None # Should be defined as a dictionary from senses to signatures (dict[string -> set[string]]) at training.
        self.word2freq = None # Should be defined as a default dictionary from words to their frequencies
        self.total_doc_num = None # Should designate the total number of documents seen during training
        self.use_idf = None # Should be a boolean depending on whether idf was learned or not

    def train(self, instances=[], window_size=-1, use_idf=False):
        """
        instances: list[WSDInstance]
        """

        self.use_idf = use_idf
        # For the signature of a sense, use (i) the definition of each of the corresponding WordNet synsets, (ii) all of the corresponding examples in WordNet and (iii) the corresponding training instances.
        self.signatures = {instance.sense: set() for instance in instances} # dictionary from senses to signatures (str->set[str])

        self.total_doc_num = 0 # variable for total number of documents
        self.word2freq = defaultdict(int) # dictionary which will store number of documents (value) in which a word (key) appears

        for instance in instances:
            # add words from context window to current words sense signature
            if window_size == -1:
                train_example_context = set(instance.context)
            else:
                train_example_context = set(instance.left_context[-window_size:] + instance.right_context[:window_size])

            self.signatures[instance.sense].update(train_example_context)

            if use_idf:
                # update document number and word doc freq
                self.total_doc_num += 1
                for word in train_example_context:
                    self.word2freq[word] += 1

            # add sense definition and examples (words from context window) to the current word sense signature
            for sense in WN_CORRESPONDANCES[instance.lemma]: # iterates over sense_id keys for words
                for synset_id in WN_CORRESPONDANCES[instance.lemma][sense]:
                    synset = wordnet.synset(synset_id)
                    definition_words = set(word_tokenize(synset.definition()))
                    self.signatures[instance.sense].update(definition_words)

                    if use_idf:
                        # update document number and word doc freq
                        self.total_doc_num += 1
                        for word in definition_words:
                            self.word2freq[word] += 1

                    # add context words from example (according to window size)
                    for example in synset.examples():
                        context_window = set(extract_context_words(word_tokenize(example), instance.lemma, window_size))
                        self.signatures[instance.sense].update(context_window)

                        if use_idf:
                            # update document number and word doc freq
                            self.total_doc_num += 1
                            for word in context_window:
                                self.word2freq[word] += 1

        # remove stop words (words with high frequency)
        for sense in self.signatures:
            self.signatures[sense] -= STOP_WORDS

    def predict_sense(self, instance):
        """
        instance: WSDInstance
        """
        
        score = {s: 0 for s in WN_CORRESPONDANCES[instance.lemma]}
        for s in score:
            for word in instance.left_context[-10:] + instance.right_context[:10]: # iterate over 10 + 10 tokens to left and right
                if word in self.signatures[s] and word not in STOP_WORDS:
                    if self.use_idf:
                        try:
                            score[s] += math.log10(self.total_doc_num/self.word2freq[word]) # calculates the inverse document frequency and takes the nat log.
                        except KeyError: # we will assume rarity if the word is not present in the training data
                            score[s] += 1
                    else:
                        score[s] += 1
        # print(instance.sense)
        # print(score)
        return max(score, key=score.get) # returns the argmax
    
    def __str__(self):
        return "SimplifiedLesk"

###############################################################



###############################################################

# The body of this conditional is executed only when this file is directly called as a script (rather than imported from another script).
if __name__ == '__main__':
    from twa import WSDCollection
    from optparse import OptionParser

    usage = "Comparison of various WSD algorithms.\n%prog TWA_FILE"
    parser = OptionParser(usage=usage)
    (opts, args) = parser.parse_args()
    if(len(args) > 0):
        sensed_tagged_data_file = args[0]
    else:
        exit(usage + '\nYou need to specify in the command the path to a file of the TWA dataset.\n')

    # Loads the corpus.
    instances = WSDCollection(sensed_tagged_data_file).instances
    
    # Displays the sense distributions.
    print("sense distributions")
    prettyprint_sense_distribution(instances)
    print("----------------------------------------------")

    # Evaluation of the random baseline on the whole corpus.
    print("random baseline accuracy on the whole corpus")
    print("%.2f" % RandomSense().evaluate(instances), "%") # truncate to only show first two decimals
    print("----------------------------------------------")
    
    # Evaluation of the most frequent sense baseline using different splits of the corpus (with `utils.data_split` or `utils.random_data_split`).
    print("most frequent baseline accuracy on a 80/20 train test data split")
    test_set, train_set = data_split(instances)
    most_freq_classifier = MostFrequentSense()
    most_freq_classifier.train(train_set)
    print("%.2f" % most_freq_classifier.evaluate(test_set), "%")
    print("----------------------------------------------")
    
    # Evaluation of Simplified Lesk (with no fixed window and no IDF values) using different splits of the corpus.
    print("simple lesk performance with no fixed window and no idf values")
    simple_lesk = SimplifiedLesk()
    simple_lesk.train(train_set)
    print("%.2f" % simple_lesk.evaluate(test_set), "%")
    print("----------------------------------------------")
    
    # Evaluation of Simplified Lesk (with a window of size 10 and no IDF values) using different splits of the corpus.
    print("simple lesk performance with a window of size 10 and no IDF values")
    simple_lesk_10 = SimplifiedLesk()
    simple_lesk_10.train(train_set, window_size=10)
    print("%.2f" % simple_lesk_10.evaluate(test_set), "%")
    print("----------------------------------------------")
    
    # Evaluation of Simplified Lesk (with IDF values and no fixed window) using different splits of the corpus.
    print("simple lesk performance with IDF values and no fixed window")
    simple_lesk_idf = SimplifiedLesk()
    simple_lesk_idf.train(train_set, use_idf=True)
    print("%.2f" % simple_lesk_idf.evaluate(test_set), "%")
    print("----------------------------------------------")
    
    # Naive Cross-validation
    avg_accuracy_simple, avg_accuracy_10, avg_accuracy_idf = [], [], []
    simple_lesk, simple_lesk_10, simple_lesk_idf = SimplifiedLesk(), SimplifiedLesk(), SimplifiedLesk()
    for i in range(100):
        # make a random split
        test_set, train_set = random_data_split(instances)
        # train each classifier and add evaluation to scoire list
        simple_lesk.train(train_set)
        avg_accuracy_simple.append(simple_lesk.evaluate(test_set))
        simple_lesk_10.train(train_set, window_size=10)
        avg_accuracy_10.append(simple_lesk_10.evaluate(test_set))
        simple_lesk_idf.train(train_set, use_idf=True)
        avg_accuracy_idf.append(simple_lesk_idf.evaluate(test_set))

    avg_accuracy_idf = sum(avg_accuracy_idf) / len(avg_accuracy_idf)
    avg_accuracy_simple = sum(avg_accuracy_simple) / len(avg_accuracy_simple)
    avg_accuracy_10 = sum(avg_accuracy_10) / len(avg_accuracy_10)

    print("cross validated accuracy on simple lesk:", "%.2f" % avg_accuracy_simple)
    print("cross validated accuracy on simple lesk with window size 10:", "%.2f" % avg_accuracy_10)
    print("cross validated accuracy on simple lesk with idf:", "%.2f" % avg_accuracy_idf)