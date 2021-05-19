import pandas as pd
import numpy as np
import sys
import re
import spacy
import string
import pickle
import datetime
import gensim
import time
import os

from pathlib import Path,PosixPath,WindowsPath,PureWindowsPath,PurePosixPath
from os.path import join

from gensim import corpora, models, similarities
from utilities import *

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation

class engine:
    '''
    A NLP process engine.
    initialization:
        + n, for LDA component number,
        + parser, for preset spacy object,
        + purge, for a dictionary of regex pattern to identify in string and purge,
        + exclusion, for a string contains all excluding punctuations and stop words,
    attributes:
        + comonent_n, an integer for LatentDirichletAllocation initial setup component number attributes
        + parser, spacy parser with en_core_web_sm as default library
        + purge, a list of text patterns for natural language purge purpose
        + exclusion, a string of characters and stopwords from spacy for exclusion purposes
        + TFIDF_core, a TFIDF object with preset parameters, is trained after data load and can be used to transform natural language text.
        + LDA_core, a LatentDirichletAllocation object with preset parameters, is trained after data load and can be used to transform natural language text. 
        + spacy_list, a complete list of sentence lists after clean up the imported data, created after load data
        + word_matrix, created after load data
        + chronicle, created after load data
        + vocab, a complete list of word vocabulary from TFIDF_core, created after load data
        + content_df, created after load data
        + word2topic_df, created after load data
        + w2v, word to vector module trained based on the cleaned out the spacy list. created after load data
    methods:
        + loadCSV
        + clean_text
        + spacy_tokenizer
        + LDA_init
        + getSimilar
        + searchKeywords
    '''

    SPACY_PARSER = spacy.load('en_core_web_sm')
    
    LDA_COMPONENT_DEFAULT = 10
    TEXT_PURGE_LS = {'return sym'   :re.compile(r'\n'),
                    'unknown char'  :re.compile(r'\x0c'),
                    'miscellaneous' :re.compile(r'[-.]'),
                    'dates'         :re.compile(r"\d+/\d+/\d+"),
                    'time'          :re.compile(r"[0-2]?[0-9]:[0-6][0-9]"),
                    'emails'        :re.compile(r"[\w]+@[\.\w]+"),
                    'websites'      :re.compile(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i")}
    TEXT_EXCLUSION = string.punctuation + str(spacy.lang.en.stop_words.STOP_WORDS)
    
    def __init__(self,
                n       = LDA_COMPONENT_DEFAULT,
                parser  = SPACY_PARSER,
                purge   = TEXT_PURGE_LS,
                exclusion = TEXT_EXCLUSION):
        self.component_n= n
        self.parser     = parser
        self.purgeLS    = purge
        self.exclusion  = exclusion
        self.TFIDF_core = TfidfVectorizer(min_df=0.0085, max_df=0.9, stop_words=ENGLISH_STOP_WORDS)
        self.LDA_core   = LatentDirichletAllocation(n_components=n, random_state=0)
    
    def loadCSV(self,fileDIR):
        '''
        load csv file into a dataframe,
        clean all excessive contents,
        
        '''
        statusUpdate('--loading CSV--')
        self.df = pd.read_csv(fileDIR, index_col = 0)
        
        statusUpdate('--converting dataframe--')
        self.df['date'] = pd.to_datetime(self.df['date'])
        art_s = self.df.groupby('filename').count()['subID'].rename('subID_count')
        art_s = art_s.loc[art_s>3]
        
        self.df = self.df.merge(art_s, on='filename', how = 'inner')
        
        statusUpdate('--created object dataframe df_subID--')
        self.df_subID = self.df[self.df['subID_count']!=0].reset_index()
        
        statusUpdate('--clean and spacy transforming dataframe content I--')
        self.text = self.df_subID['content']
        self.text = self.text.apply(lambda x: self.clean_text(x))
        self.text = self.text.apply(lambda x: ' '.join(self.spacy_tokenizer(x)))
        
        statusUpdate('--clean and spacy transforming dataframe content II--')
        clean_list = [self.clean_text(i) for i in self.text]
        self.spacy_list = [self.spacy_tokenizer(i) for i in clean_list]
        
        statusUpdate('--get word matrix and vocabulary--')
        self.word_matrix = self.TFIDF_core.fit_transform(self.text)
        self.vocab = self.TFIDF_core.get_feature_names()
        #print(len(self.vocab))
        
        statusUpdate('--initializing LDA core--')
        self.content_df, self.word2topic_df = self.LDA_init()
    
        statusUpdate('--initializing word to vector core--')
        self.w2v = gensim.models.Word2Vec(self.spacy_list, size=100, window=5, min_count=1, workers=2, sg=1)
        self.chronicle = self.df['date'].unique()
    
    def getText(self,filename = None):
        if filename is None:
            text = ''.join(self.df['content'].tolist())
        else:
            text = ''.join(self.df[self.df['filename']==filename]['content'].tolist())
        text.replace('\n',' ')
        return text
    
    def clean_text(self,text):
        '''
        '''
        for k,p in self.purgeLS.items():
            text = re.sub(p,' ',text)
            pure_text = ''
        # Validate to check if there are any non-text content 
        for letter in text:
            # Keep only letters and spaces
            if letter.isalpha() or letter==' ':
                pure_text += letter
        # Join the words are not stand-alone letters
        text = ' '.join(word for word in pure_text.split() if len(word)>1)
        return text
    
    # Creating our tokenizer function
    def spacy_tokenizer(self,text):
        # Creating our token object, which is used to create documents with linguistic annotations.
        mytokens = self.parser(text)
        # Lemmatizing each token and converting each token into lowercase
        mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
        # Removing stop words
        mytokens = [ word for word in mytokens if word not in self.exclusion ]
        # return preprocessed list of tokens
        return mytokens

    def LDA_init(self):
        try:
            self.LDA_core.fit(self.word_matrix)
        except Exception as err:
            print(err)
            return
        
        topic_matrix = self.LDA_core.transform(self.word_matrix)
        topic_matrix_df = pd.DataFrame(topic_matrix).add_prefix('topic_')
        topic_matrix_df["topic"] = topic_matrix_df.iloc[:,:].idxmax(axis=1)
        content_df = pd.concat([self.df_subID,topic_matrix_df], axis=1)
        
        word2topic_df = pd.DataFrame(self.LDA_core.components_, columns=self.vocab).T.add_prefix('topic_')
        return content_df, word2topic_df

    def getSimilar(self,keywords,topn = 20):
        if self.content_df is None:
            statusUpdate('--engine is not initialized--')
        try:
            result = [w[0] for w in self.w2v.wv.most_similar(keywords,topn = topn)]
        except Exception as err:
            print(err)
            print("Try Another Word")
        
        for i in range(topn):
            try:
                result.append(self.w2v.wv.most_similar(keywords,topn = topn)[i][0])
            except Exception as err:
                print(err)
                print("Try Another Word")
        return result
    
    def searchKeywords(self,keyWords,timeA,timeB,hasDollar = False):
        # preprocess all the input variables
        if type(keyWords) == type(' '):
            keyWords = keyWords.split(',')
        elif type(keyWords)==type([]):
            pass
        else:
            pass
        
        startdate = timeA
        enddate = timeB
        
        # generate the similar keywords list
        '''
        keyword_list=[]
        for i in range(20):
            try:
               keyword_list.append(self.w2v.wv.most_similar(keywords ,topn=20)[i][0])
            except:
                pass
        '''
        
        # check and remove excessive error words for further processing.
        pattern_err_word = re.compile(r"word '(.*)' not in vocabulary")
        keyword_list = None
        
        while(keyword_list is None and keyWords != []):
            try:
                keyword_list = [k for (k,v) in self.w2v.wv.most_similar(keyWords ,topn=20)]+keyWords
            except KeyError as err:
                err_word = re.search(pattern_err_word,str(err)).group(1)
                keyWords.remove(err_word)

            finally:
                if keyWords == []:
                    return None

        # generate the topic list as per keyword in keyword list
        topic_list =[]
        for k in keyword_list:
            try:
                topic_list.append(self.word2topic_df.loc[k].idxmax())
            except:
                pass
                
        time_frame = (self.content_df['date'] > startdate) & (self.content_df['date'] < enddate)
        sub_df = self.content_df[time_frame]
        if hasDollar == True:
            sub_df = sub_df[sub_df['hasDollar']==True]

        notes = pd.DataFrame()

        for j in range(len(list(set(topic_list)))): # number of unique topics

            n = 0
            for i in range(len(topic_list)):

                if list(set(topic_list))[j] == topic_list[i]:
                    n = n+3  # Number of notes can be controled to show 

            notes = pd.concat([notes, sub_df.sort_values(list(set(topic_list))[j], ascending = False)[0:n]])
            
        return notes, topic_list, keyword_list
    
    def updateContent(self,filename):
        pass
        
    def removeContent(self,filename):
        pass
        
    def viewContent(self,filename):
        return self.df[self.df['filename']==filename]
    

    def addContent(self,filename):
        new_df = pdf2df(filename)
        
        statusUpdate('--converting dataframe--')
        new_df['date'] = pd.to_datetime(new_df['date'])
        #print(new_df.head())
        #print(new_df.count()[])
        
        art_i = new_df.count()['date']
        new_df['subID_count'] = art_i
        
        self.df = self.df.append(new_df,ignore_index = True)
        
        statusUpdate('--created object dataframe df_subID--')
        newSub_df = new_df[new_df['subID_count']!=0].reset_index()
        self.df_subID = self.df_subID.append(newSub_df,ignore_index = True)
        
        statusUpdate('--clean and spacy transforming dataframe content I--')
        text = newSub_df['content']
        text = text.apply(lambda x: self.clean_text(x))
        self.text.append(text.apply(lambda x: ' '.join(self.spacy_tokenizer(x))),ignore_index = True)
        
        statusUpdate('--clean and spacy transforming dataframe content II--')
        clean_list = [self.clean_text(i) for i in text]
        self.spacy_list += [self.spacy_tokenizer(i) for i in clean_list]
        self.chronicle = self.df['date'].unique()
        
        statusUpdate('--retrain engine--')
        if self.retrain():
            statusUpdate('--engine retrained--')
        else:
            statusUpdate('--engine retrain failed--')
        
        

# get number of pages from the full content base
    def getPageCount(self):
        pageNo_pattern = re.compile(r'Page\s+(\d+)\s')
        self.df['pageNo'] = self.df['content'].apply(lambda x: re.search(pageNo_pattern,x) is not None)
        subdf = self.df[self.df['pageNo']].copy()
        subdf['pageNo'] = subdf['content'].apply(lambda x: re.search(pageNo_pattern,x).group(1)).astype('int32')
        return subdf.groupby('filename')['pageNo'].max().sum()
    
    def getWordCount(self):
        text = [c['content'] for i,c in self.df.iterrows()]
        return sum([len(s.split(' ')) for s in text])
    
    def getOriginal(self,filename):
        file_df = self.df[self.df['filename']==filename].copy()
        text = ''.join([c['content'] for i,c in file_df.iterrows()])
        return text
    
    def retrain(self):
        try:
            print('engine retrain started')
            self.text.reset_index()
            self.word_matrix = self.TFIDF_core.fit_transform(self.text)
            self.vocab = self.TFIDF_core.get_feature_names()
            print('TFIDF core training complete')
            self.w2v = gensim.models.Word2Vec(self.spacy_list, size=100, window=5, min_count=1, workers=2, sg=1)
            print('word to vector core training complete')
        except Exception as err:
            print(err)
            return False
        
        try:
            self.content_df, self.word2topic_df = self.LDA_init()
            print('LDA core training complete')
        except Exception as err:
            print(err)
            return False
        return True

    
def testinit(filename = 'city_sanjose_data.csv'):
    start = time.time()
    Engine = engine()
    Engine.loadCSV(filename)
    end = time.time()
    
    print(f'takes {end-start} second to train')
    return Engine

def saveEngine(filename, obj_engine):
    
    start = time.time()
    fulldir = join(os.getcwd(),filename)
    file_to_open = Path(fulldir)
    pickle.dump(obj_engine,file_to_open.open('wb'))
        
    end = time.time()
    print(f'takes {end-start} second to save')
    
    return filename

def loadEngine(filename):
    start = time.time()
    fulldir = join(os.getcwd(),filename)
    file_to_open = Path(fulldir)
    obj_engine = pickle.load(file_to_open.open('rb'))
        
    end = time.time()
    print(f'takes {end-start} second to load')
    return obj_engine

def test():
    statusUpdate('--test start--')
    enginefilename = 'nlp_engine.pkl'

    init_engine = testinit()
    
    saved_name = savepkl(enginefilename,init_engine)
    copy_engine = loadpkl(saved_name)
    
    statusUpdate('--test end--')
    statusUpdate(f'--test saved file name: {saved_name}--')

if __name__ == '__main__':
    # User Inputs
    test()
