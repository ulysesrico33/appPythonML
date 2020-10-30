import MLfunctions as mlf
from gensim import corpora
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import gensim
import seaborn as sns
import matplotlib.colors as mcolors



sw=stopwords.words('spanish')
pathtohere=os.getcwd()


def main():
    print('LSI model with gensim')
    print('1) 1 gram, 2) 2 gram, 3) 3 gram')
    op=input()
    op=int(op)
    lsReturn=[]
    lsDocuments=[]
    lsSubject=[]
    #Get the the information into a list of documents
    lsReturn=mlf.getRawTextToList()
    lsDocuments=lsReturn[0]
    lsSubject=lsReturn[1]
    lsDocuments_NoSW = [[word for word in simple_preprocess(str(doc)) if word not in sw] for doc in lsDocuments]
    if(op==1):
        print('LSI model with gensim for 1 gram')
        
    if(op==2):
        print('LSI model with gensim for 2 gram')
        bigram = gensim.models.Phrases(lsDocuments_NoSW, min_count=5, threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        lsDocBiGram = [bigram_mod[doc] for doc in lsDocuments_NoSW]
        lsDocuments_NoSW.clear()
        lsDocuments_NoSW = [[word for word in simple_preprocess(str(doc)) if word not in sw] for doc in lsDocBiGram]
          

    if(op==3):
        print('LSI model with gensim for 3 gram')
        bigram = gensim.models.Phrases(lsDocuments_NoSW, min_count=5, threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram = gensim.models.Phrases(bigram[lsDocuments_NoSW], threshold=100)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        lsDocTrigram = [trigram_mod[doc] for doc in lsDocuments_NoSW]
        lsDocuments_NoSW.clear()
        lsDocuments_NoSW = [[word for word in simple_preprocess(str(doc)) if word not in sw] for doc in lsDocTrigram]
        
        """
        print('Getting bigrams list...')
        for doc in lsDocuments_NoSW:
            for word in doc:
                mlf.appendInfoToFile(pathtohere,'\\trigrams.txt',word+'\n')
        """        
          

    print('LSI Model starting...')
    # Create Dictionary
    id2word = corpora.Dictionary(lsDocuments_NoSW)
    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in lsDocuments_NoSW]
    # Build LDA model
    lsi_model = gensim.models.LsiModel(corpus=corpus,
                                id2word=id2word,
                                num_topics=20)
    
    df=pd.DataFrame()
    #df=mlf.getDominantTopicDataFrame(lsi_model,corpus,lsDocuments_NoSW,lsSubject)  
    #mlf.generateFileSeparatedBySemicolon(df,'LSI_trigram_csv.txt')    

    lsi_cm=CoherenceModel(model=lsi_model,corpus=corpus,dictionary=id2word,texts=lsDocuments_NoSW)
    print('LSI Coherence:',lsi_cm.get_coherence())                       
                                                        
 

if __name__=='__main__':
    main()    
