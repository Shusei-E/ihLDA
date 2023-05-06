# dataread.pyx
"""Read data from files
"""

import os
import numpy as np
from statistics import median
from cpython cimport bool
cimport numpy as np
cimport cython

cdef extern from "math.h":
   double log(double x)

ctypedef np.int32_t dtype_t
ctypedef np.float64_t ftype_t

cdef class DICTIONARY:
    """
    Read the file and create corpus dictionary
    """

    cdef dict wordid_to_word
    cdef dict word_to_wordid
    cdef list doc_word_frequency  # frequency of words for each doc_id
    cdef dict word_frequency
    cdef dict word_ids_in_doc  # unique words in doc
    cdef dict docs_contain_wordid  # documents that have a certain word_id
    cdef int num_vocab

    # temp use
    cdef list text_wordid
    cdef list text_wordid_test
    cdef dict temp_d_w_freq
    cdef list temp_list
    cdef int word_id
    cdef int doc_id
    cdef list documents
    cdef int count

    def __cinit__(self):
        self.wordid_to_word = {}
        self.word_to_wordid = {}
        self.doc_word_frequency = []
        self.word_frequency = {}
        self.word_ids_in_doc = {}
        self.docs_contain_wordid = {}
        self.num_vocab = 0

    def read_file(self, file_path, doc_id):
        """Read files without splitting

        It uses :code:`cdef process_file()` to read a file.

        1. Open the file
        2. Read words one by one
        3. Convert each word to word_id (:code:`get_wordid()`)
        4. Update :code:`self.word_frequency` (dict)

        Args:
            * file_path (str)
            * doc_id (int)

        Returns:
            self.text_wordid (list): a list of word ids in a text

        """

        return self.process_file(file_path, doc_id)

    def read_file_split(self, file_path, doc_id):
        """
        Split corpus into training and test sets

        Each word in each document is randomly assigned\
                to training set or test set.

        It uses :code:`cdef process_file_split()` to read a file.

        1. Open the file
        2. Read words one by one
        3. By 20% chance, it stores the word_id to\
                :code:`self.text_wordid_test` (list).\
                Be careful that it stores **raw words**.
        4. Convert each word to word_id (:code:`get_wordid()`)
        5. Update :code:`self.word_frequency` (dict)

        Args:
            * file_path (str)
            * doc_id (int)

        Returns:
            self.text_wordid (list): a list of word ids in a text
        """

        return self.process_file_split(file_path, doc_id)

    def get_variables(self):
        """
        Return important data

        Returns:
            * self.wordid_to_word
            * self.word_to_wordid
            * self.word_frequency
            * self.word_ids_in_doc
            * self.docs_contain_wordid
            * self.num_vocab
        """

        return (self.wordid_to_word, self.word_to_wordid, self.word_frequency,
                self.word_ids_in_doc, self.docs_contain_wordid, self.num_vocab)

    def get_doc_word_frequency(self, list documents):
        """

        Read each document and create an object to store\
                document-word frequency.

        Args:
            * documents (list): a list of documents in word_id format
            * verbose (bool, default=False): if there are ten or more\
                    documents, it will automatically set\
                    :code:`verbose = True`.

        Returns:
            numpy arrays in a list
        """

        self.create_doc_word_frequency(documents)
        return self.doc_word_frequency

    cdef create_doc_word_frequency(self, list documents_in, bool verbose=False):
        print("Organizing documents...")
        self.documents = documents_in

        total_num = len(self.documents)
        self.count = 0
        threshold = int(total_num/10)
        percentile = 10
        if total_num > 10:
            verbose = True

        self.doc_id = 0
        for self.text_wordid in self.documents:
            self.temp_d_w_freq = {}  # clear

            for self.word_id in self.text_wordid:  # for each document
                if self.word_id in self.temp_d_w_freq:
                    self.temp_d_w_freq[self.word_id] += 1
                else:
                    self.temp_d_w_freq[self.word_id] = 1

            self.temp_list = []
            for self.word_id in range(self.num_vocab):
                if self.word_id in self.word_ids_in_doc[self.doc_id]:
                    self.temp_list.append(self.temp_d_w_freq[self.word_id])
                else:
                    self.temp_list.append(0)

            self.doc_word_frequency.append(np.array(self.temp_list))

            self.count += 1
            if threshold < self.count and verbose:
                print_sentence = "     Organizing documents: {percent}% finished"
                print(print_sentence.format(percent=str(percentile)))
                percentile += 10
                self.count = 0

            self.doc_id += 1

    cdef list process_file(self, str file_path, int doc_id):
        self.text_wordid = []  # clear
        self.word_ids_in_doc[doc_id] = set()  # insert empty set

        with open(file_path, "r") as f:
            whole_text = f.read()
            text = whole_text.split()

            for word in text:
                self.word_id = self.get_wordid(word)
                self.text_wordid.append(self.word_id)
                self.word_ids_in_doc[doc_id].add(self.word_id)

                if self.word_id in self.word_frequency:
                    self.word_frequency[self.word_id] += 1
                else:
                    self.word_frequency[self.word_id] = 1

                if self.word_id in self.docs_contain_wordid:
                    self.docs_contain_wordid[self.word_id].add(doc_id)
                else:
                    self.docs_contain_wordid[self.word_id] = set()
                    self.docs_contain_wordid[self.word_id].add(doc_id)

        return self.text_wordid

    cdef tuple process_file_split(self, str file_path, int doc_id):
        self.text_wordid = []  # clear
        self.text_wordid_test = []
        self.word_ids_in_doc[doc_id] = set()  # insert empty set

        with open(file_path, "r") as f:
            whole_text = f.read()
            text = whole_text.split()

            for word in text:
                if 0.8 < np.random.uniform(0.0, 1.0):
                    self.text_wordid_test.append(word)  # Be careful this is a raw word
                    continue

                self.word_id = self.get_wordid(word)
                self.text_wordid.append(self.word_id)
                self.word_ids_in_doc[doc_id].add(self.word_id)

                if self.word_id in self.word_frequency:
                    self.word_frequency[self.word_id] += 1
                else:
                    self.word_frequency[self.word_id] = 1

                if self.word_id in self.docs_contain_wordid:
                    self.docs_contain_wordid[self.word_id].add(doc_id)
                else:
                    self.docs_contain_wordid[self.word_id] = set()
                    self.docs_contain_wordid[self.word_id].add(doc_id)

        return (self.text_wordid, self.text_wordid_test)

    cdef int get_wordid(self, str word):
        if word in self.word_to_wordid:  # check key of dictionry
            return self.word_to_wordid[word]

        else:
            word_id = self.num_vocab
            self.word_to_wordid[word] = word_id
            self.wordid_to_word[word_id] = word

            self.num_vocab += 1

            return word_id


class DATA_READ:
    """Python-readable class to read data

    Args:
        * files_list (list): a list of files to read
        * train_test_split (bool):
                whether or not split documents\
                into a test set and a train set. You need to set\
                the percentage manually in :code:`process_file_split()`.

    Important objets:
        * self.documents (list):
            contains lists of wordids for each document (training set)
        * self.documents_test (list):
            contains lists of wordids for test set if you split corpus
        * self.each_doc_len (int)
        * self.total_len (int)
        * self.docid_to_filename (dict)
        * self.filename_to_docid (dict)
    """

    def __init__(self, list files_list, bool train_test_split):
        """Initialization
        """

        self.files_list = files_list
        self.doc_num = len(self.files_list)  # Number of documents
        self.train_test_split = train_test_split
        self.doc_len_median = 0

        self.dict = DICTIONARY()

        self.read_data()
        self.integrate()

    def integrate(self):
        """
        Get information from instantiated DICTIONARY class and\
        delete DICTIONARY class object.
        """

        # Cythonized Class only used for reading documents
        (self.wordid_to_word, self.word_to_wordid,
         self.word_frequency, self.word_ids_in_doc,
         self.docs_contain_wordid, self.num_vocab) = self.dict.get_variables()

        # self.doc_word_frequency = self.dict.get_doc_word_frequency(self.documents)

        del self.dict

    def read_data(self, bool verbose=True):
        """Read data main function

        Args:
            * verbose (bool): whether or not show progress.\
                    If the number of documents is fewer than\
                    10, it will automatically set\
                    :code:`verbose = False`.

        Returns:
            None
        """
        self.documents = []  # contains lists of wordids for each document
        self.documents_test = []  # contains lists of wordids for each document
        self.each_doc_len = []  # length of each document
        self.total_len = 0  # total length of corpus
        self.docid_to_filename = {}
        self.filename_to_docid = {}

        print("Loading documents...")
        total_num = len(self.files_list)
        count = 0
        threshold = int(total_num/10)
        percentile = 10
        if total_num < 10:
            verbose = False

        doc_id = 0
        for file_path in self.files_list:
            if self.train_test_split:
                # Split
                obj = self.dict.read_file_split(file_path, doc_id)  # Read file here
                text_wordid = obj[0]
                text_wordid_test = obj[1]
                self.documents_test.append(text_wordid_test)
            else:
                # Do not split
                text_wordid = self.dict.read_file(file_path, doc_id)  # Read file here

            self.docid_to_filename[doc_id] = os.path.basename(file_path)
            self.filename_to_docid[file_path] = doc_id

            self.documents.append(text_wordid)
            doc_len = len(text_wordid)
            self.each_doc_len.append(doc_len)
            self.total_len += doc_len

            doc_id += 1

            count += 1
            if threshold < count and verbose:
                print_str = "     Loading documents: {percent}% finished, {doc_id} docs"
                print(print_str.format(percent=str(percentile), doc_id=str(doc_id)))
                percentile += 10
                count = 0

    def get_doc_num(self):
        """Number of documents

        Args:
            None

        Returns:
            self.doc_num (int)
        """

        return self.doc_num

    def get_doclen_median(self):
        """Median of document length
        """

        if self.doc_len_median == 0:
            self.doc_len_median = median(self.each_doc_len)

        return self.doc_len_median

    def get_g0(self):
        """Base probability of words

        Args:
            None
        Returns:
            base probaility of trained words (np.array)
        """

        g0 = np.array([self.word_frequency[word_id]
                      for word_id in range(self.num_vocab)]) / self.total_len
        return g0

    def get_logg0(self):
        """Base probability of words in log

        Args:
            None
        Returns:
            base probaility of trained words in log (np.array)
        """

        logg0 = np.array([log(self.word_frequency[word_id])
                         for word_id in range(self.num_vocab)]) - log(self.total_len)
        return logg0

    def get_word_from_wordid(self, wordid):
        """Search a word that corresponds to a wordid

        Args:
            wordid (int)
        Returns:
            word (str)
        """

        return self.wordid_to_word[wordid]

    def get_wordid_from_word(self, word):
        """Search a word that corresponds to a wordid

        Args:
            wordid (int)
        Returns:
            word (str)
        """

        try:
            return self.word_to_wordid[word]
        except KeyError:
            return ""

    def get_filename_from_docid(self, docid):
        """Search a original filename that corresponds to docid

        Args:
            docid (int)
        Returns:
            filename (str)
        """

        return self.docid_to_filename[docid]
