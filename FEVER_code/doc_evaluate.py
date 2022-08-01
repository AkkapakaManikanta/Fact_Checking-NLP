import os
import pandas as pd
import ast
import pickle

data_path = os.path.dirname(os.path.abspath(''))

def document_retrieval_evaluate():

    csv_data = pd.read_csv(os.path.join(data_path, "FEVER_data/shared_task_dev.csv"))
    csv_data['evidence'] = csv_data['evidence'].apply(lambda x: ast.literal_eval(x))
    
    claims = csv_data.claim.values
    evidences = csv_data.evidence.values
    labels = csv_data.label.values
    verifiable = csv_data.verifiable.values
    
    with open('results/documents_dev.pickle', 'rb') as f:
        documents = pickle.load(f)
  
    total = len(documents)
    
    k = 0
    for i in range(total):
        found = True
        if verifiable[i] == 'VERIFIABLE':
            urls = documents[i]
            for j in evidences[i]:
                if len(j) == 1:
                    found = False
                    if j[0][2].replace('-COLON-', ':').replace('-RRB-', ')').replace('-LRB-', '(') in urls:
                        found = True
                        break
        if found:
            k += 1
    return k / total * 100

print("The accuracy of prediction of the document retrieval phase is:" ,document_retrieval_evaluate())
