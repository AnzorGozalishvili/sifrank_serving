#!/usr/bin/env python

import nltk
from flask import Flask, jsonify, request
import stanza
from embeddings import sent_emb_sif, word_emb_elmo
from model.method import SIFRank, SIFRank_plus
import os

app = Flask(__name__)


@app.route('/sifrank')
def sifrank():
    query = request.args.get('q')
    top_n = int(request.args.get('n', 15))
    keywords = SIFRank(query, SIF, en_model, N=top_n, elmo_layers_weight=elmo_layers_weight)

    return jsonify(keywords)


@app.route('/sifrankplus')
def sifrankplus():
    query = request.args.get('q')
    top_n = int(request.args.get('n', 15))
    keywords = SIFRank_plus(query, SIF, en_model, N=top_n, elmo_layers_weight=elmo_layers_weight)

    return jsonify(keywords)


if __name__ == '__main__':
    # download from https://allennlp.org/elmo
    options_file = "/data/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "/data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    cuda_device_id = int(os.environ.get('NVIDIA_VISIBLE_DEVICES', 0))

    porter = nltk.PorterStemmer()
    ELMO = word_emb_elmo.WordEmbeddings(options_file, weight_file, cuda_device=cuda_device_id)
    SIF = sent_emb_sif.SentEmbeddings(ELMO, lamda=1.0)
    en_model = stanza.Pipeline(lang='en', processors={}, use_gpu=True)
    elmo_layers_weight = [0.0, 1.0, 0.0]

    app.run(host='0.0.0.0', port=5000)
