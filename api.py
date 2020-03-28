#!/usr/bin/env python

import nltk
import stanza
from flask import Flask, jsonify, request

from embeddings import sent_emb_sif, word_emb_elmo
from model.method import SIFRank, SIFRank_plus

app = Flask(__name__)


@app.route('/sifrank', methods=['POST'])
def sifrank():
    data = request.json
    query = data.get("text", "")
    top_n = int(data.get("n", 15))

    keywords = SIFRank(query, SIF, en_model, N=top_n, elmo_layers_weight=elmo_layers_weight)

    return jsonify(keywords)


@app.route('/sifrankplus', methods=['POST'])
def sifrankplus():
    data = request.json
    query = data.get("text", "")
    top_n = int(data.get("n", 15))

    keywords = SIFRank_plus(query, SIF, en_model, N=top_n, elmo_layers_weight=elmo_layers_weight)

    return jsonify(keywords)


if __name__ == '__main__':
    data_dir = "data/"
    options_file = f"{data_dir}elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = f"{data_dir}elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    cuda_device = 0

    porter = nltk.PorterStemmer()
    ELMO = word_emb_elmo.WordEmbeddings(options_file=options_file, weight_file=weight_file, cuda_device=cuda_device)
    SIF = sent_emb_sif.SentEmbeddings(word_embeddor=ELMO, data_dir=data_dir, lamda=1.0)
    en_model = stanza.Pipeline(lang='en', processors={}, use_gpu=True)
    elmo_layers_weight = [0.0, 1.0, 0.0]

    app.run(host='0.0.0.0', port=5000)
