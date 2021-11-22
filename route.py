from flask import Flask, render_template, request, redirect, url_for
import json, random, pickle, models, torch, pandas as pd, data, tqdm

device = 'cpu'
if torch.cuda.is_available():
    torch.device('cuda:0')
else:
    torch.device('cpu')

model = torch.load("infer/model.pt", map_location=torch.device('cpu'))
train_vocab = data.JSONDataset.load("infer/train_dataset.pt").vocab

print(type(model))
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
   app.run(port=5000)
   # load_bigger_graph()