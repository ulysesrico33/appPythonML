from io import StringIO
import nltk
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import os
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyLDAvis.gensim
import gensim
from datetime import datetime
from gensim.models import CoherenceModel
nltk.download('stopwords')
nltk.download('punkt')
pathtohere=os.getcwd()
sw=stopwords.words('spanish')


def convertListToString(lst):
    print('Converting Listo to String...')
    strDoc=StringIO()
    for doc in lst:
        strDoc.write(str(doc)+' ')
    return strDoc.getvalue()


def clean_corpus(words,sw):
    print('Cleaning list of Documents: Getting rid of puntuaction and stopwords...')
    words_no_pun=[]
    for w in words:
        if w.isalpha():
            words_no_pun.append(w.lower())
    #Remove stopwords
    clean_words=[]
    for w in words_no_pun:
        if w not in sw:
            clean_words.append(w)
    return clean_words    

def get_TFIDF():
    print('Getting TF-IDF matrix...')
    lsDocuments=[]
    lsDocuments=getCorpusList()
    tfidf=TfidfVectorizer(encoding='utf-8',stop_words=sw,smooth_idf=True)
    lsReturn=[]
    lsReturn.append(tfidf)
    lsReturn.append(ltDocuments)
    return lsReturn

def getCountVectorizer():
    
    lsDocuments=[]
    lsDocuments=getCorpusList()
    count_vect = CountVectorizer(stop_words=sw)
    lsReturn=[]
    lsReturn.append(count_vect)
    lsReturn.append(ltDocuments) 

    return lsReturn

def getRawTextToList():
    print('Getting information from database into a python list...')
    cloud_config= {

    'secure_connect_bundle': pathtohere+'//secure-connect-dbquart.zip'
         
    }
    
    objCC=CassandraConnection()
    auth_provider = PlainTextAuthProvider(objCC.cc_user,objCC.cc_pwd)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    session = cluster.connect()
    row=''
    ltDocuments=[]
    lsSubject=[]
    #lsNoThesis=[]
    querySt="select heading,text_content,subject,type_of_thesis from thesis.tbthesis where period_number=10 ALLOW FILTERING"       
    statement = SimpleStatement(querySt, fetch_size=1000)
    print('Getting data from datastax...')
    for row in session.execute(statement):
        thesis_b=StringIO()
        #Add subject to a list aside
        lsSubject.append(row[2])
        #lsNoThesis.append(row[4])
        for col in row:
            if type(col) is list:
                for e in col:
                    thesis_b.write(str(e)+' ')
            else:        
                thesis_b.write(str(col)+' ')
        thesis=''
        thesis=thesis_b.getvalue()
        ltDocuments.append(thesis)
    lsReturn=[]    
    lsReturn.append(ltDocuments)
    lsReturn.append(lsSubject)
    #lsReturn.append(lsNoThesis)
    return lsReturn

def appendInfoToFile(path,filename,strcontent):
    txtFile=open(path+filename,'a+')
    txtFile.write(strcontent)
    txtFile.close()

def getDominantTopicDataFrame(lda_model,corpus,lsDocuments_NoSW,lsSubject):
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row in enumerate(lda_model[corpus]):           
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = lda_model.show_topic(topicid=topic_num,topn=2000)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break


    # Add original text to the end of the output
    #'contents' is created as a new set of rows (Series) which can be added later 
    contents = pd.Series(lsDocuments_NoSW)
    subject=pd.Series(lsSubject)
    #concat([A,B]) is actually adding another column, hence is correct, so from 3 columns, it ends up with 4
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    sent_topics_df = pd.concat([sent_topics_df, subject], axis=1)
    sent_topics_df.columns = ['Topic_No', 'Dominant_Topic', 'Keywords','Text','Subject'] 

    return sent_topics_df 

def generateFileSeparatedBySemicolon(sent_topics_df,fileName):
    fileContent=''
    fileContent=sent_topics_df.columns[0]+';'+sent_topics_df.columns[1]+';'+sent_topics_df.columns[2]+';'+sent_topics_df.columns[3]+';'+sent_topics_df.columns[4]+'\n'
    appendInfoToFile(pathtohere,'\\'+fileName,fileContent)
    for index, row in sent_topics_df.iterrows():
        fileContent=str(row['Topic_No'])+' ;'+str(row['Dominant_Topic'])+' ;'+str(row['Keywords'])+' ;'+str(row['Text'])+' ;'+str(row['Subject'])+'\n'
        appendInfoToFile(pathtohere,'\\'+fileName,fileContent)  


def generateGraphDWC(sent_topics_df):
    doc_lens = [len(d) for d in sent_topics_df.Text]
    # Plot
    plt.figure(figsize=(16,7), dpi=160)
    plt.hist(doc_lens, bins = 1000, color='navy')
    plt.text(750, 100, "Mean   : " + str(round(np.mean(doc_lens))))
    plt.text(750,  90, "Median : " + str(round(np.median(doc_lens))))
    plt.text(750,  80, "Stdev   : " + str(round(np.std(doc_lens))))
    plt.text(750,  70, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
    plt.text(750,  60, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

    plt.gca().set(xlim=(0, 1000), ylabel='Number of Documents', xlabel='Document Word Count')
    plt.tick_params(size=16)
    plt.xticks(np.linspace(0,1000,9))
    plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
    plt.savefig(pathtohere+'\\wordsSpreadInAllDoc.png')
    plt.show()

def generatePyLDAVis(lda_model,corpus,fileName):
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    pyLDAvis.save_html(vis,fileName)    


def readFile(file):
    ls=[]
    file1 = open(file, 'r',encoding='utf8') 
    Lines = file1.readlines() 
    for line in Lines: 
        ls.append(line.strip())   
    return ls 


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    appendInfoToFile(pathtohere+'\\','scores.txt','No.Topic,Coherence Score\n')
    for num_topics in range(start, limit, step):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=num_topics)
        model_list.append(lda_model)
        coherencemodel = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        appendInfoToFile(pathtohere+'\\','scores.txt',str(num_topics)+','+str(coherencemodel.get_coherence())+'\n')

    return model_list, coherence_values 

def getTime():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time      


         
    
class CassandraConnection():
    cc_user='quartadmin'
    cc_pwd='P@ssw0rd33'        