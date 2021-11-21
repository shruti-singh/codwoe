from flask import Flask, render_template, request, redirect, url_for
import json
import pandas as pd
import random
import pickle

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

if __name__ == '__main__':
   app.run(port=5000)
   # load_bigger_graph()