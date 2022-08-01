import ast
import csv
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data_path = os.path.dirname(os.path.abspath(''))
wiki_path = os.path.join(data_path, "FEVER_data/wiki_pages.csv")

w = {}

def wiki_pages():
    global w
    data = pd.read_csv(wiki_path)  #nrows=70000
    w = dict()
    for r in data.iterrows():
        w[r[1]['id']] = r[1]['text']   #put the id - title and the text of that document into the dictionary w.
    del data
  
vectorizer = TfidfVectorizer(ngram_range=(1, 1), lowercase=True, max_df=0.85, binary=True) # best paremeters are chosen.

def similar_sentences_in_document(claim, title):
 
    text = w[title].replace('  ', '  .  .').split(' . ')  #Get the text of that title from w and split it by full stops.

    vector = vectorizer.fit_transform([claim] + text)  #text is the splitted version of the text of the document for claim. Give the claim and the lines on sentence as input.
    #print(tfidf)
    similarity = cosine_similarity(vector[0], vector).flatten()[1:] #Find the similarities of claim to all others
    #print(similarity)
    return similarity  #so, this returns the list of cosine sim on claim to all sentences in doc.


def similar_sentences(urls,claim,top):
    #print('Im finding sent for' + claim )
    #print(urls)
    result = {}
    
    sims = np.array([])
    lens = [0]
    titles = []
    
    for title in urls: #Iterate over the titles to search for. Basically tuples.
        title = title.replace('(', '-LRB-').replace(')', '-RRB-').replace(':' , '-COLON-')#Basic replacement of the titles
        try:
            similarities = similar_sentences_in_document(claim, title)  #Finds the documents and their similarities
            sims = np.concatenate([sims, similarities])  
            #print(sims)
            lens.append(len(similarities) + lens[-1])
            titles.append(title)
        except:
            continue
    
    lens = np.array(lens)
    for ind in sims.argsort()[-top:]:  #Since we need only the top 5 results, we pick the first 5
        res = lens[lens - ind <= 0].argmax()
        result.setdefault(titles[res], dict())
        result[titles[res]][ind - lens[res]] = sims[ind]
    #print(result)
    return result

def save_results(claims, documents,task_type,top=5):
    #print('Im at claims ret')
    saver = []
    for i in range(len(claims)):
        claim = claims[i]  #For every claim in the data, for every relevant doc, find the sentences.
        urls = documents[i]   #Here the urls will be tuple of values. Thats how its stored in the dr phase.
        saver.append(similar_sentences(urls,claim, top))  #urls - tuple, claim - sentence
    
    with open('results/results_{}.pickle'.format(task_type), 'wb') as f:
        pickle.dump(saver, f)


def create_bert_file(task_type, claims,top=5):

    #print('Im at bert ret')
    if w == {}:
        wiki_pages()  #get the title and text into w as dictionary from wiki pages csv.
        
    with open('results/documents_{}.pickle'.format(task_type), 'rb') as f:  #Get the relevant documents titles of the claim.
        documents = pickle.load(f)
    
    save_results(claims, documents,task_type,top)  #Find the sr result.
    with open('results/results_{}.pickle'.format(task_type), 'rb') as f:
        saver = pickle.load(f)
    
    with open('results/pred_{}.tsv'.format(task_type), "w", encoding="utf-8") as file:
        f = csv.writer(file, delimiter='\t')
        f.writerow(["title_text", "claim", "title", "sent", "index"])
        for i in range(len(saver)):
            new_result = {}
            result = saver[i]
            claim = claims[i]
            
            for title in result:
                for sent in result[title]:
                    new_result[(title, sent)] = result[title][sent]
            
            for j in list(sorted(new_result.items(), key=lambda x: -x[1])):
                doc = j[0][0]
                t = doc.replace('_', ' ')
                sent = j[0][1]
                text = w[doc].replace('  ', '  .  .').split(' . ')
                
                if text[sent] != '' and text[sent].strip()[0] == '.':
                    text[sent] = text[sent].strip()[1:]
                title_text = '# ' + t + ' # ' + text[sent] + ' . '
                f.writerow([title_text, claim, doc, sent, i])

                
def sentence_retrieval(task_type, claims=None, top=5):
    
    #Sentence retrieval is done only for non-training data as it is meaningless to do on the train data.
    
    if task_type == 'dev':
        train = pd.read_csv(os.path.join(data_path, "FEVER_data/shared_task_dev.csv"))
    if task_type == 'test':
        train = pd.read_csv(os.path.join(data_path, "FEVER_data/shared_task_test.csv"))
    claims = train.claim.values
    
    #print('Im at sent ret')
    create_bert_file(task_type, claims,top)  #Creating the bert file of the relevant sentences to test for further support/refute.
                
                
def train_bert_file(task_type, top=5):
    
    wiki_pages()  #Loads the wiki pages path. Puts the title and the text into w.
        
    with open('results/documents_{}.pickle'.format(task_type), 'rb') as f:  #It has the index and the corresponding heading of titlees.
        documents = pickle.load(f)  #Results of the documents retrieved as dict.
        
    if task_type == 'train':
        csv_data = pd.read_csv(os.path.join(data_path, "FEVER_data/train.csv"))  #CSV file to get the lines.
    if task_type == 'dev':
        csv_data = pd.read_csv(os.path.join(data_path, "FEVER_data/shared_task_dev.csv")) 
           
    csv_data['evidence'] = csv_data['evidence'].apply(lambda x: ast.literal_eval(x))
    claims = csv_data.claim.values  #Has all the claims
    evidences = csv_data.evidence.values #Has all the evidences of the csv
    labels = csv_data.label.values  #corresponding labels supports or reject.
    verifiable = csv_data.verifiable.values  #verificable facts.

    with open('results/{}.tsv'.format(task_type), "w", encoding="utf-8") as file:  #Basically we are creating a file with the evidence line file name, and the claim for the bert.
        f = csv.writer(file, delimiter='\t')
        f.writerow(["title_text", "claim", "label"])
        
        for i in range(len(claims)):
            if verifiable[i] == 'VERIFIABLE':   #Provided that the claim is a verifiable one,
                for j in evidences[i]:   #iterate the evidences of that claim.
                    if len(p) == 1:   #Having a single evidence for the claim.
                        try:
                            text = w[j[0][2]].replace('  ', '  .  .').split(' . ') #j[0][2] is title . So, get the text of doc from w.
                            title = j[0][2].replace('_', ' ').split('-LRB-')[0]
                            title_text = '# ' + title + ' # ' + text[j[0][3]] + '.'  #j[0][3] is the sentence number in the doc.So ,we are adding the corresponding line evidence.
                            claim = claims[i]  #Claim to be written.
                            label = labels[i]   #Whether the sentence is supporting.
                            f.writerow([title_text, claim, label])
                            break
                        except:
                            continue
                            
            else:  #If the claim is of the case which cannot be verified, 
                #we are using the docs since we took the docs for the non-verifiable stuff.
                result = similar_sentences(documents[i],claims[i],top)  #Need to find the sentences in the doc similar to claim.
                claim = claims[i]  #The claim for consideration.
                label = labels[i]   #Label of the claim.
                for doc in result:  # A claim can have several documents tp search for. So iterate. doc is headings or titles.
                    t = doc.replace('_', ' ').split('-LRB-')[0]
                    for sent in result[doc]:  #Now, get the sentences of that doc.
                        text = w[doc].replace('  ', '  .  .').split(' . ')   #Find the text and replace.
                        title_text = '# ' + t + ' # ' + text[sent] + '.' #Writes the sentences with the claim.
                        f.writerow([title_text, claim, label])

    csv_data = pd.read_csv('results/{}.tsv'.format(task_type), sep='\t')
    csv_data= csv_data.sample(frac=1, random_state=41).reset_index(drop=True)
    csv_data.to_csv('results/{}.tsv'.format(task_type), sep='\t', index=False)


def sentence_retrieval_evaluate(top=5):
    train = pd.read_csv(os.path.join(data_path, "FEVER_data/shared_task_dev.csv"))
    train['evidence'] = train['evidence'].apply(lambda x: ast.literal_eval(x))
    
    claims = train.claim.values
    evidences = train.evidence.values
    labels = train.label.values
    verifiable = train.verifiable.values
    
    with open(os.path.join(data_path, 'FEVER_code/results/documents_dev.pickle'), 'rb') as f:
        documents = pickle.load(f)
   
   
    wiki_pages()
    
    k = 0
    for i in range(len(claims)):
        flag = True
        if verifiable[i] == 'VERIFIABLE':
            claim = claims[i]
            urls = documents[i]
            result = similar_sentences(urls,claim,top)
            for j in evidences[i]:
                if len(j) == 1:
                    flag = False
                    if j[0][3] in result.get(j[0][2], []):
                        flag = True
                        break
        if flag:
            k += 1
    
    return k/len(claims) * 100

print("The accuracy of the model for sentence retrieval is " , sentence_retrieval_evaluate())
#print(w)
#train_bert_file('train', 5)
#claims = ['Sancho Panza is a fictional character in a novel written by a Spanish writer born in the 17th century.','Alexander Pushkin was born in Paris.']
#sentence_retrieval('dev')
