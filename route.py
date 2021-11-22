from flask import Flask, render_template, request, redirect, url_for
import json
import pandas as pd
import plotly
import plotly.express as px
import random
import pickle

from plotly.offline import plot
from plotly.graph_objs import Scatter

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

# @app.route('/graph_demo',methods=['POST','GET'])
# def data():
# 	dataa=request.args.get('data')
# 	return redirect('graph')

# @app.route('/graph',methods = ['POST', 'GET'])
# def graph():
	
# 	nodeid = request.args.get('data')
# 	# graph_found = testvis(nodeid)
# 	graph_found = render_subgraph_for_paper(nodeid)
# 	if not graph_found:
# 		print("Not enough information about the citation network of the paper in the dataset!!!")
# 	return render_template("paper_search.html",a1=[])
# 	# return render_template('sigma.js/examples/drag-nodes.html')
# 	# return render_template('gameofthrones.html')

@app.route('/team',methods=['POST','GET'])
def team():
	return render_template("team.html")


@app.route('/gen_gloss',methods=['POST','GET'])
def gen_gloss():
	if request.method == "POST":
		first_name = request.form.get("emb-text")
		last_name = request.form.get("gloss-text")
		print(first_name, last_name)
	return render_template("gen_gloss.html")

if __name__ == '__main__':
   app.run(port=5000)
   # load_bigger_graph()