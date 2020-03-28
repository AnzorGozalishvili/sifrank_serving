## SIFRank serving

see original paper [SIFRank: A New Baseline for Unsupervised Keyphrase Extraction Based on Pre-trained Language Model](https://ieeexplore.ieee.org/document/8954611)

# Getting Started

## docker-compose parameters
Flask api running on port 5000 will be mapped to outer 5001 port.
(it uses docker-compose version 2.3 which supports `runtime: nvidia` to easily use GPU environment inside container)
```yaml
version: '2.3'
services:
  sifrank_serving:
    container_name: sifrank_serving
    build: .
    ports:
      - "5001:5000"
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

## Running Docker Container
Probably the easiest way to get started is by using the provided Docker image.
From the project's root directory, the image can be built and running like so:
```
$ docker-compose up --build -d
```
This can take a several minutes to finish. All model files and everything will be downloaded automatically.

## Flask API (inside container)
This return for each text a tuple containing three lists:
1) The top N candidates (string) i.e keyphrases
2) For each keyphrase the associated relevance score

```python
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
```

## Curl example to call API

You can simply query from terminal
```bash
curl --location --request POST 'http://0.0.0.0:5000/sifrank' \            
    --header 'Content-Type: application/json' \
    --data-raw '{"text": "A New Baseline for Unsupervised Keyphrase Extraction Based on Pre-trained Language Model.", "n":10}'
    
curl --location --request POST 'http://0.0.0.0:5000/sifrankplus' \            
    --header 'Content-Type: application/json' \
    --data-raw '{"text": "A New Baseline for Unsupervised Keyphrase Extraction Based on Pre-trained Language Model.", "n":10}'
```

Response
```bash

```

## POST request from python
```python
import json
import requests

sifrank_url = f"http://0.0.0.0:5000/sifrank"
sifrank_plus_url = f"http://0.0.0.0:5000/sifrankplus"

data = {"text": "A New Baseline for Unsupervised Keyphrase Extraction Based on Pre-trained Language Model.", "lang":"en", "n":10}
sifrank_result = requests.post(sifrank_url, json=data)
sifrank_plus_result = requests.post(sifrank_plus_url, json=data)

sifrank_content = json.loads(sifrank_result.content)
sifrank_plus_content = json.loads(sifrank_plus_result.content)

sifrank_keywords = sifrank_content[0]
sifrank_relevance_scores = sifrank_content[1]

sifrank_plus_keywords = sifrank_plus_content[0]
sifrank_plus_relevance_scores = sifrank_plus_content[1]
```

## Local Run

## requirements
```json
allennlp==0.9.0
stanza==1.0.0
```
## Download Models
* ELMo ``elmo_2x4096_512_2048cnn_2xhighway_options.json`` (already downloaded in `data/`) and 
``elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5`` from [here](https://allennlp.org/elmo) , and save it to the `data/` directory

## Sample usage
```python
import nltk
from embeddings import sent_emb_sif, word_emb_elmo
from model.method import SIFRank, SIFRank_plus
import stanza

# download from https://allennlp.org/elmo
options_file = "data/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

porter = nltk.PorterStemmer()
ELMO = word_emb_elmo.WordEmbeddings(options_file, weight_file, cuda_device=0)
SIF = sent_emb_sif.SentEmbeddings(ELMO, lamda=1.0)
en_model = stanza.Pipeline(lang='en', processors={}, use_gpu=True)
elmo_layers_weight = [0.0, 1.0, 0.0]

text = "Discrete output feedback sliding mode control of second order systems - a moving switching line approach The sliding mode control systems (SMCS) for which the switching variable is designed independent of the initial conditions are known to be sensitive to parameter variations and extraneous disturbances during the reaching phase. For second order systems this drawback is eliminated by using the moving switching line technique where the switching line is initially designed to pass the initial conditions and is subsequently moved towards a predetermined switching line. In this paper, we make use of the above idea of moving switching line together with the reaching law approach to design a discrete output feedback sliding mode control. The main contributions of this work are such that we do not require to use system states as it makes use of only the output samples for designing the controller. and by using the moving switching line a low sensitivity system is obtained through shortening the reaching phase. Simulation results show that the fast output sampling feedback guarantees sliding motion similar to that obtained using state feedback"
keyphrases = SIFRank(text, SIF, en_model, N=15, elmo_layers_weight=elmo_layers_weight)
keyphrases_ = SIFRank_plus(text, SIF, en_model, N=15, elmo_layers_weight=elmo_layers_weight)
print(keyphrases)
print(keyphrases_)
```

## SIFRank evaluation results from original [repository](https://github.com/sunyilgdx/SIFRank)
**F1 score** when the number of keyphrases extracted N is set to 5.

| Models       | Inspec       | SemEval2017   | DUC2001      |
| :-----       | :----:       | :----:        |:----:        |
| TFIDF        | 11.28        | 12.70         |  9.21        |
| YAKE         | 15.73        | 11.84         | 10.61        |
| TextRank     | 24.39        | 16.43         | 13.94        |
| SingleRank   | 24.69        | 18.23         | 21.56        |
| TopicRank    | 22.76        | 17.10         | 20.37        |
| PositionRank | 25.19        | 18.23         | 24.95        |
| Multipartite | 23.05        | 17.39         | 21.86        |
| RVA          | 21.91        | 19.59         | 20.32        |
| EmbedRank d2v| 27.20        | 20.21         | 21.74        |
| SIFRank      | **29.11**        | **22.59**         | 24.27        |
| SIFRank+     | 28.49        | 21.53         | **30.88**        |

## Cite
If you use this code, please cite this paper
```
@article{DBLP:journals/access/SunQZWZ20,
  author    = {Yi Sun and
               Hangping Qiu and
               Yu Zheng and
               Zhongwei Wang and
               Chaoran Zhang},
  title     = {SIFRank: {A} New Baseline for Unsupervised Keyphrase Extraction Based
               on Pre-Trained Language Model},
  journal   = {{IEEE} Access},
  volume    = {8},
  pages     = {10896--10906},
  year      = {2020},
  url       = {https://doi.org/10.1109/ACCESS.2020.2965087},
  doi       = {10.1109/ACCESS.2020.2965087},
  timestamp = {Fri, 07 Feb 2020 12:04:22 +0100},
  biburl    = {https://dblp.org/rec/journals/access/SunQZWZ20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
