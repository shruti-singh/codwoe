from flask import Flask, render_template, request, redirect, url_for
import json
import pandas as pd
import random

import pickle
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
stop_words = set(stopwords.words('english'))

import models
import torch
import json
import tqdm
import data
from sklearn.decomposition import PCA

device = 'cpu'
if torch.cuda.is_available():
    torch.device('cpu')
else:
    torch.device('cpu')
pca = PCA(n_components=2)
model = models.RevdictModel.load("D:\\nlp613\\codwoe_app\\infer\\model.pt").to('cpu')
train_vocab = data.JSONDataset.load("D:\\nlp613\\codwoe_app\\infer\\train_dataset.pt").vocab

geeky_file = open('D:\\nlp613\\codwoe_app\\pcaweights.pkl', 'rb')
weights = pickle.load(geeky_file)
geeky_file.close()  

pca.set_params(**weights)

app = Flask(__name__)

l_color = {"INT": "red", "REL": "blue", "DATA": "yellow", "MET": "cyan", "RES": "black"}

@app.route('/',methods = ['POST', 'GET'])
def student():
	return render_template('index.html')

@app.route('/analysis',methods=['POST','GET'])
def analysis():
	return render_template("emb_neigh_analysis.html")

@app.route('/compare_emb',methods=['POST','GET'])
def compare_emb():
	print(request.form)
	if 'favorite_pet' in request.form:
		print(request.form['favorite_pet'])
		print(request.form.get('Birds'))
		print(request.form.get('Dogs'))
		print(request.form.get('Cats'))
	else:
		print(request.form)
		#form_data = 
	return render_template("compare_emb.html")

@app.route('/paper_search',methods=['POST','GET'])
def result():
	if 'search-text' in request.form:
		print(request.form['search-text'])
		dataa = search_es_for_papers(request.form['search-text'])
		return render_template("paper_search.html",a1=dataa,v=request.form['search-text'])
	else:
		return render_template("paper_search.html",a1=[])

@app.route('/sandbox',methods=['POST','GET'])
def sandbox():
	sen = request.form.get("input-text")
	sen = str(sen)
	w = sen.split()
	random.shuffle(w)
	stext=' '.join(w)
	utext = sen.upper()
	sen_tokens = word_tokenize(sen)
	wl = [wo for wo in sen_tokens if not wo in stop_words]
	srtext = ' '.join(wl)
	tagged = nltk.pos_tag(word_tokenize(sen))
	tag_words=[]
	for tup in tagged:
		if tup[1] == 'NN':
			pass
		else:
			tag_words.append(tup)
	rnn = ' '.join(tag_words[i][0] for i in range(len(tag_words)))
	test_file = r"D:\nlp613\codwoe_app\modelip.json"
	pred_file = r"D:\nlp613\codwoe_app\modelop.json"

	#with open(test_file, "w") as ostr:
	#	json.dump([{"id": "xyz", "gloss": sen}], ostr)
	userip = sen
	userip2 = srtext
	#3.NN
	userip3 = rnn
	#4.shuffle
	userip4 = stext
	#5.uppercase
	userip5 = utext
	glosses = [userip,userip2,userip3,userip4,userip5]
	glosdicts = []
	i=0
	for glos in glosses:
		i=i+1  
		glosdicts.append({"id": str(i), "gloss": glos})
	with open(test_file, "w") as ostr:
		json.dump(glosdicts, ostr)
	target_arch = 'sgns'
	test_dataset = data.JSONDataset(
		test_file, vocab=train_vocab, freeze_vocab=True, maxlen=model.maxlen
	)
	test_dataloader = data.get_dataloader(test_dataset, shuffle=False, batch_size=1024)
	vec_tensor_key = f"{target_arch}_tensor"
	assert test_dataset.has_gloss, "File is not usable for the task"
	# 2. make predictions
	predictions = []
	with torch.no_grad():
		pbar = tqdm.tqdm(desc="Pred.", total=len(test_dataset))
		for batch in test_dataloader:
			#print(batch["gloss_tensor"])
			vecs = model(batch["gloss_tensor"].to('cpu')).to('cpu')
			for id, vec in zip(batch["id"], vecs.unbind()):
				predictions.append(
					{"id": id, target_arch: vec.view(-1).tolist()}
				)
			pbar.update(vecs.size(0))
		pbar.close()
	print(weights, "\n", predictions)
	#embed_pca = predictions
	o = []
	for lis in predictions:
		o.append(lis['sgns'])
	embed_pca = pca.fit_transform(o)
	#print(embed_pca)
	return render_template("sandbox.html", complete = sen, stext = stext, utext=utext, srtext = srtext, rnn= rnn, embed_pca = embed_pca)

@app.route('/team',methods=['POST','GET'])
def team():
	return render_template("team.html")

if __name__ == '__main__':
   app.run(port=5001, debug=True, use_debugger=False)
   # load_bigger_graph()