import MLfunctions as mlf
import gensim
from gensim import corpora
from gensim.models import Doc2Vec,Word2Vec,TfidfModel
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.colors as mcolors

sw=stopwords.words('spanish')
pathtohere=os.getcwd()


def main():
    print('LDA model with gensim')
    print('1) 1 gram, 2) 2 gram, 3) 3 gram, 4) Ranking of coherence')
    op=input()
    op=int(op)
    #28 topics, optimum result (27 topics are really 28, 0 to 27)
    numberTopic=5
    lsReturn=[]
    lsDocuments=[]
    lsSubject=[]
    #lsNoThesis=[]
    #Get the the information into a list of documents
    lsReturn=mlf.getRawTextToList()
    lsDocuments=lsReturn[0]
    lsSubject=lsReturn[1]
    #lsNoThesis=lsReturn[2]
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
        print('LDA model with gensim for 1 gram')
        
    if(op==2):
        print('LDA model with gensim for 2 gram')
        bigram = gensim.models.Phrases(lsDocuments_NoSW, min_count=5, threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        lsDocBiGram = [bigram_mod[doc] for doc in lsDocuments_NoSW]
        lsDocuments_NoSW.clear()
        lsDocuments_NoSW = [[word for word in simple_preprocess(str(doc)) if word not in sw] for doc in lsDocBiGram]
          

    if(op==3):
        print('LDA model with gensim for 3 gram')
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

    if (op==4):
        print('Starting coherence ranking with 2 gram...')  
        #Generate best coherence ranking
        # Create Dictionary
        id2word = corpora.Dictionary(lsDocuments_NoSW)
        # Create Corpus: Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in lsDocuments_NoSW]
        bigram = gensim.models.Phrases(lsDocuments_NoSW, min_count=5, threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        lsDocBiGram = [bigram_mod[doc] for doc in lsDocuments_NoSW]
        lsDocuments_NoSW.clear()
        lsDocuments_NoSW = [[word for word in simple_preprocess(str(doc)) if word not in sw] for doc in lsDocBiGram]
        limit=51; start=2; step=1;
        model_list, coherence_values = mlf.compute_coherence_values(dictionary=id2word, corpus=corpus, texts=lsDocuments_NoSW, start=start, limit=limit, step=step)
        print('Plotting ranking...')
        # Show graph
        
        x = range(start, limit, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show() 
        sys.exit()

    # id2word :Create Dictionary, this dictionary has the id and word
    
    id2word = corpora.Dictionary(lsDocuments_NoSW)

    # Term Document Frequency
    #Gensim creates a unique id for each word in the document. 
    #The produced corpus shown above is a mapping of (word_id, word_frequency).
    
    corpus = [id2word.doc2bow(text) for text in lsDocuments_NoSW]
    #Example: it has 37, 342 indexes, so 0 to 37, 341 
    columns=len(id2word) 
    #Generate list of columns
    lsColumn=[]
    for i in range(0,columns):
        lsColumn.append(str(i));
    #Generate the indexes (id_thesis)
    lsIndex=[]    
    lsIndex=mlf.readFile('lsThesis.txt')
    term_matrix=[]
    lim=columns-1
    countDoc=0
    for doc in corpus:
        strdoc=''
        for i in range(0,columns): 
            bFound=False 
            for index_word,value in doc: 
                #Case: When the document has that index word 
                if int(i)!=lim:
                    if i==index_word: 
                        bFound=True 
                    if i==0:
                        strdoc='('
                    if int(index_word)==i:
                        strdoc=strdoc+str(value)+',' 
                    break    
                #Case: End of columns then add value and ')'    
                else:    
                    if i==index_word: 
                        bFound=True 
                        if int(index_word)==i:
                            strdoc=strdoc+str(value)+')' 
                        break    
            if bFound==False:
                strdoc=strdoc+'0,' 
            if bFound==False and i==lim:
                strdoc=strdoc+'0)'

        #mlf.appendInfoToFile(pathtohere+'\\','vectors.txt',strdoc)                                    
        term_matrix.append(strdoc)
        countDoc=countDoc+1
        print('Doc:',str(countDoc))    
   

    dataFrame = pd.DataFrame(term_matrix) 
    for row in dataFrame.iterrows():
        mlf.appendInfoToFile(pathtohere+'\\','dataFrameContent.txt',str(row))

    sys.exit()

    #Print the id and word 
    """
     for element in lsSubject:
        mlf.appendInfoToFile(pathtohere+'\\','lsSubject.txt',str(element)+'\n')    
     
    #Get the word and its ID.
    for key,value in id2word.token2id.items():
        mlf.appendInfoToFile(pathtohere+'\\','id2word.txt',str(key)+';'+str(value)+'\n')    

    """    
        
    print('LDA Model starting...')
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                id2word=id2word,
                                num_topics=numberTopic)

    """
    print('Printing topics')
    lda_topics=lda_model.print_topics()
    for topic in lda_topics:
        mlf.appendInfoToFile(pathtohere,'\\list_of_topics_lda.txt',str(topic)+'\n')
    """
    
    df=pd.DataFrame()
    df=mlf.getDominantTopicDataFrame(lda_model,corpus,lsDocuments_NoSW,lsSubject)  
    mlf.generateFileSeparatedBySemicolon(df,str(op)+'gram_csv_'+str(numberTopic)+'_withoutCompleteList.txt')                          
                                                        
    mlf.generatePyLDAVis(lda_model,corpus,'vis_'+str(op)+'gram_'+str(numberTopic)+'_withoutCompleteList.html')
    
    """
    lda_cm=CoherenceModel(model=lda_model,corpus=corpus,dictionary=id2word,texts=lsDocuments_NoSW)
    print('LDA Coherence:',lda_cm.get_coherence())    
    """


if __name__=='__main__':
    main()    
