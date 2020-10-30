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
    print('HDP model with gensim')
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
    #Read the unwanted words and then add them up to stopwords
    lsUnWantedWords=[]
    lsUnWantedWords=mlf.readFile('removed_words.txt')
    for word in lsUnWantedWords:
        sw.append(word.strip())
        
    #Read the Notsure words and then add them up to stopwords
    lsNotSureWords=[]
    lsNotSureWords=mlf.readFile('notsure_words.txt')
    for word in lsNotSureWords:
        sw.append(word.strip())
       
    lsDocuments_NoSW = [[word for word in simple_preprocess(str(doc)) if word not in sw] for doc in lsDocuments]
    if(op==1):
        print('HDP model with gensim for 1 gram')
        
    if(op==2):
        print('HDP model with gensim for 2 gram')
        bigram = gensim.models.Phrases(lsDocuments_NoSW, min_count=5, threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        lsDocBiGram = [bigram_mod[doc] for doc in lsDocuments_NoSW]
        lsDocuments_NoSW.clear()
        lsDocuments_NoSW = [[word for word in simple_preprocess(str(doc)) if word not in sw] for doc in lsDocBiGram]
          

    if(op==3):
        print('HDP model with gensim for 3 gram')
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
          

    print('HDP Model starting...')
    # Create Dictionary
    id2word = corpora.Dictionary(lsDocuments_NoSW)
    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in lsDocuments_NoSW]
    # Build HDP model
    hdp_model = gensim.models.hdpmodel.HdpModel(corpus=corpus,id2word=id2word)
    
    print('Printing topics')
    hdp_topics=hdp_model.print_topics()
    for topic in hdp_topics:
        mlf.appendInfoToFile(pathtohere,'\\list_of_topics_hdp_'+str(op)+'gram_RemovedWords_NotSure.txt',str(topic)+'\n')
   

    """
    HDP no genera esta función, no tiene un parámetro necesario que sí tiene LDA
    Para saber el no. de tópicas usar la función de arriba
    df=pd.DataFrame()
    df=mlf.getDominantTopicDataFrame(hdp_model,corpus,lsDocuments_NoSW,lsSubject)  
    mlf.generateFileSeparatedBySemicolon(df,'HDP_'+str(op)+'gram_csv.txt') 
    """
    
    """ 
    hdp_cm=CoherenceModel(model=hdp_model,corpus=corpus,dictionary=id2word,texts=lsDocuments_NoSW)
    print('HDP Coherence:',hdp_cm.get_coherence())       
    """                     
                                                        


if __name__=='__main__':
    main()    
