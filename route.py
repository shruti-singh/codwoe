from flask import Flask, render_template, request, redirect, url_for
import json, random, pickle, models, torch, pandas as pd, data, tqdm
import json
import pandas as pd
import plotly
import plotly.express as px
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

import plotly
import plotly.express as px
from plotly.offline import plot
from plotly.graph_objs import Scatter

device = 'cpu'
if torch.cuda.is_available():
    torch.device('cpu')
else:
    torch.device('cpu')
pca = PCA(n_components=2)
model = models.RevdictModel.load(r".\infer\model.pt").to('cpu')
train_vocab = data.JSONDataset.load(r".\infer\train_dataset.pt").vocab

geeky_file = open(r'.\infer\pcaweights.pkl', 'rb')
weights = pickle.load(geeky_file)
geeky_file.close()  

pca.set_params(**weights)

from plotly.offline import plot
from plotly.graph_objs import Scatter

device = 'cpu'
if torch.cuda.is_available():
    torch.device('cuda:0')
else:
    torch.device('cpu')

model = torch.load("infer/model.pt", map_location=torch.device('cpu'))
train_vocab = data.JSONDataset.load("infer/train_dataset.pt").vocab

print(type(model))
app = Flask(__name__)

word_pca_data = json.load(open('data/word_pca_new.json'))

l_color = {"INT": "red", "REL": "blue", "DATA": "yellow", "MET": "cyan", "RES": "black"}

@app.route('/',methods = ['POST', 'GET'])
def student():
	return render_template('index.html')

@app.route('/analysis',methods=['POST','GET'])
def analysis():
	return render_template("emb_neigh_analysis.html")

@app.route('/compare_emb',methods=['POST','GET'])
def compare_emb():
	if 'dword[]' in request.form:
		dict_words = request.form.getlist('dword[]')
	else:
		dict_words = ['mafia', 'liar', 'dead']
	rows_list = []
	for arch in ['sgns', 'char', 'elec']:
		for w in dict_words:
			rows_list.append({'word': w, 'arch': arch, 'x': 0, 'y': 0})
			rows_list.append({'word': w, 'arch': arch, 'x': word_pca_data[w][arch][0], 'y': word_pca_data[w][arch][1]})
	df = pd.DataFrame(rows_list)
	fig_sgns = px.line(df.loc[df['arch'] == 'sgns'], x='x', y='y', color='word')
	fig_sgns.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
	fig_char = px.line(df.loc[df['arch'] == 'char'], x='x', y='y', color='word')
	fig_char.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
	fig_electra = px.line(df.loc[df['arch'] == 'elec'], x='x', y='y', color='word')
	fig_electra.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))

	graphJSON_sgns = json.dumps(fig_sgns, cls=plotly.utils.PlotlyJSONEncoder)
	graphJSON_char = json.dumps(fig_char, cls=plotly.utils.PlotlyJSONEncoder)
	graphJSON_electra = json.dumps(fig_electra, cls=plotly.utils.PlotlyJSONEncoder)

	return render_template('compare_emb.html', graphJSON_sgns=graphJSON_sgns, graphJSON_char=graphJSON_char, graphJSON_electra=graphJSON_electra)
	#return render_template("compare_emb.html")


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
	test_file = r".\infer\modelip.json"
	pred_file = r".\infer\modelop.json"

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
	target_arch_all = ['sgns', 'char', 'electra']
	for target_arch in target_arch_all:
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
		o = []
		for lis in predictions:
			o.append(lis[target_arch])
		embed_pca = pca.fit_transform(o)
		embed_df = pd.DataFrame(data = embed_pca, columns = ['pc1', 'pc2'])
		embed_df['glos'] = glosses
		fig_sgns = px.line(embed_df, x=embed_df['pc1'], y = embed_df['pc2'], color='glos')
		fig_char = px.line(embed_df, x=embed_df['pc1'], y = embed_df['pc2'], color='glos')
		fig_electra = px.line(embed_df, x=embed_df['pc1'], y = embed_df['pc2'], color='glos')

	graphJSON_sgns = json.dumps(fig_sgns, cls=plotly.utils.PlotlyJSONEncoder)
	graphJSON_char = json.dumps(fig_char, cls=plotly.utils.PlotlyJSONEncoder)
	graphJSON_electra = json.dumps(fig_electra, cls=plotly.utils.PlotlyJSONEncoder)
	return render_template("sandbox.html", complete = sen, stext = stext, utext=utext, srtext = srtext, rnn= rnn, embed_pca = embed_pca, graphJSON_sgns=graphJSON_sgns , graphJSON_char=graphJSON_char, graphJSON_electra=graphJSON_electra)

@app.route('/team',methods=['POST','GET'])
def team():
	return render_template("team.html")


@app.route('/gen_gloss',methods=['POST','GET'])
def gen_gloss():
	if request.method == "POST":
		first_name = request.form.get("emb-text")
		last_name = request.form.get("gloss-text")
		print(first_name, last_name)

	test_file = "infer/en.test.json"
	test_dataset = data.JSONDataset(
        test_file, vocab=train_vocab, freeze_vocab=True, maxlen=model.maxlen
    )
	test_dataloader = data.get_dataloader(test_dataset, shuffle=False, batch_size=1024)
	source_arch = "electra"
	vec_tensor_key = f"{source_arch}_tensor"
	if source_arch == "electra":
		assert test_dataset.has_electra, "File is not usable for the task"
	else:
		assert test_dataset.has_vecs, "File is not usable for the task"
	
	
	# 2. make predictions
	predictions = []
	with torch.no_grad():
		pbar = tqdm.tqdm(desc="Pred.", total=len(test_dataset), disable=None)
		for batch in test_dataloader:
			sequence = model.pred(batch[vec_tensor_key].to("cpu"))
			for id, gloss in zip(batch["id"], test_dataset.decode(sequence)):
				predictions.append({"id": id, "gloss": gloss})
			pbar.update(batch[vec_tensor_key].size(0))
		pbar.close()
    # 3. dump predictions
	print(predictions[:5])
    # with open(args.pred_file, "a") as ostr:
    #     json.dump(predictions, ostr)

	return render_template("gen_gloss.html")

if __name__ == '__main__':
   app.run(port=5001, debug=True, use_debugger=False)
   # load_bigger_graph()