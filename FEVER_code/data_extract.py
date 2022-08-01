import json 
import csv 
import io
import os
import pandas as pd

def extract_json(fileobj): 
    with fileobj: 
        for line in fileobj: 
            yield json.loads(line)
            

data_dir = os.path.dirname(os.path.abspath(''))


def extract_data(json_path,output_csv, task):
	
	if task != 'wikipages':
		json_data = io.open(json_path, mode='r', encoding='utf-8')
		extracted_data = extract_json(json_data) 
	
	f = csv.writer(open(output_csv, "w", encoding="utf-8"))
	if task == 'train' or task == 'dev':
		f.writerow(["id", "verifiable", "label", "claim", "evidence"])
		for r in extracted_data:
			f.writerow([r["id"], r["verifiable"], r["label"], r["claim"], r["evidence"]])
	elif task == 'test':
		f.writerow(["id", "claim"])
		for r in extracted_data:
   			f.writerow([r["id"], r["claim"]])
	else:
		f.writerow(["id", "text", "lines"])
		for file in os.listdir(json_path):
			json_data = io.open(json_path + file, mode='r', encoding='utf-8')
			extracted_data = extract_json(json_data) 
			for r in extracted_data:
				f.writerow([r["id"], r["text"], r["lines"]])


if __name__ == "__main__":
	extract_data(os.path.join(data_dir, 'FEVER_data/train.jsonl'), os.path.join(data_dir, "FEVER_data/train.csv"), 'train')
	extract_data(os.path.join(data_dir, 'FEVER_data/shared_task_dev.jsonl'), os.path.join(data_dir, "FEVER_data/shared_task_dev.csv"), 'dev')
	extract_data(os.path.join(data_dir, 'FEVER_data/shared_task_test.jsonl'), os.path.join(data_dir, "FEVER_data/shared_task_test.csv"), 'test')
	extract_data(os.path.join(data_dir, 'FEVER_data/wiki-pages/'), os.path.join(data_dir, "FEVER_data/wiki_pages.csv"), 'wikipages')


	train = pd.read_csv(os.path.join(data_dir, "FEVER_data/train.csv"))
	dev = pd.read_csv(os.path.join(data_dir, "FEVER_data/shared_task_dev.csv")) 
	test = pd.read_csv(os.path.join(data_dir, "FEVER_data/shared_task_test.csv"))
