# Preprocess the data
print("Loading Libraries...")
import glob
import re
import os
import time
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
from nltk import stem
from nltk.corpus import stopwords
import random
from gensim import corpora

class Preprocess:
    def __init__(self, folder, save_folder, random_pick=-1, prune=True):
        self.folder = folder
        self.save_folder = save_folder
        self.files_list = glob.glob(self.folder + "/*.txt") + glob.glob(self.folder + "/*/*.txt")
        self.prune = prune

        self.make_save_folder_empty()

        if random_pick > 0:
            # randomly choose files
            print("Randomly choose {Num} files.".format(Num=str(random_pick)))
            self.files_list = random.sample(self.files_list, random_pick)
        self.read_doc_num = len(self.files_list)

        self.prepare()
        self.read_files()

        if self.prune:
            self.corpus.filter_extremes(no_below=self.no_below, no_above=self.no_above)
            self.save_corpus()

    def make_save_folder_empty(self):
        save_folder_files = glob.glob(self.save_folder + "/*.txt")
        if len(save_folder_files) > 0:
            ans = ""
            while ans != "y" and ans != "n":
                ans = input("Do you want to clear the save folder? (y/n): ")

            if ans == "y":
                for file_ in save_folder_files:
                    os.remove(file_)

            time.sleep(2.5)

    def prepare(self):
        ### prepare stemmer
        self.stemmer = EnglishStemmer()

        ### prepare lemmatizer
        self.lemmatizer = stem.WordNetLemmatizer() # lemmatizer.lemmatize('word')

        ### stopwords list
        self.stopset = set(stopwords.words("english"))

        self.corpus = corpora.Dictionary()
        self.prune_at = 2000000
        self.save_file_path = [] # Order same as corpus documents
        self.documents = []

        # filter parameter
        self.no_below = int(self.read_doc_num * 0.02)
        self.no_above = 0.99

        # Other variables
        self.total_doc = 0

    def read_files(self):
        print("File processing started: {Total} files in total".format(Total=str(len(self.files_list))))
        self.total_doc = 0
        for file_path in self.files_list:
            self.preprocess(file_path)
            self.total_doc += 1
            if self.total_doc % 100 == 0:
                print("   Finished Reading: {Num} files".format(Num=str(self.total_doc)))

    def save_corpus(self):
        # Check whether or not to save
        print(self.corpus)
        get_line = ""
        while get_line != "y" and get_line != "n":
            get_line = input("Do you want to save files? (y/n): ")
        if get_line == "n":
            import sys ; sys.exit()


        counter = 0
        percentile = int(self.total_doc / 10)
        finished_per = 10

        for doc_id in range(len(self.save_file_path)):
            save_path = self.save_file_path[doc_id]

            text = [ self.corpus[bow[0]] for bow in self.corpus.doc2bow( self.documents[doc_id] ) for x in range(bow[1]) ]
            self.save_text(save_path, text)

            counter += 1
            if counter >= percentile:
                print("   Saved: {per}% of total files".format(per=str(finished_per)))
                counter = 0
                finished_per += 10


    def save_text(self, save_path, list_text):
        text = " ".join(list_text)
        if len(text) == 0:
            print("text length is 0, skip")
            return
        with open(save_path, "w") as f:
            f.write(text)


    def preprocess(self, file_path):
        with open(file_path, "r") as f:
            try:
                text = f.read()
            except:
                print("File Skipped: reading error in", file_path)
                return

            # lower cases
            text = text.lower()

            # tokenize on spaces
            text = CountVectorizer().build_tokenizer()(text)

            # delete numbers
            text = [word for word in text if re.search("\d", word) == None]
            text = [word for word in text if re.search("_", word) == None]

            # delete stopwords
            text = [word for word in text if word not in self.stopset]

            # stem / lemmatize
            text = [self.lemmatizer.lemmatize(word) for word in text] # lemmatize

            # length of the word
            text = [word for word in text if len(word) > 2]


        if len(text) == 0:
            print("Empty file:", file_path)
            return

        save_path = self.save_folder + os.path.basename(file_path)
        self.save_file_path.append(save_path)

        if self.prune:
            self.corpus.add_documents([text], prune_at = self.prune_at)
            self.documents.append(text)
            return
        else:
            self.save_text(self, save_path, text)
            return


if __name__ == '__main__':
    folder = "./input/sample_raw/"
    save_folder = "./input/sample/"
    random_pick = -1 # all -> -1

    preprocess = Preprocess(folder, save_folder, random_pick)

    print("Finished.")