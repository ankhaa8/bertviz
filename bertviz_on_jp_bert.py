from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

from bertviz.attention_details import AttentionDetailsData, show
from bertviz.pytorch_pretrained_bert import BertModel, BertTokenizer
print("imported the libraries!")
### Model dir
f = pd.read_csv("/Users/bayartsogtyadamsuren/Downloads/bert-japanese-files/bertviz_samples/webdentsuho_text_category_20190620.csv")

### Output path
ff = open("/Users/bayartsogtyadamsuren/Downloads/bert-japanese-files/bertviz_samples/bert_viz_samples.tsv", "w")

def _get_attention_details(tokens_a, tokens_b, query_vectors, key_vectors, atts):
    key_vectors_dict = defaultdict(list)
    query_vectors_dict = defaultdict(list)
    atts_dict = defaultdict(list)

    slice_a = slice(0, len(tokens_a))  # Positions corresponding to sentence A in input
    slice_b = slice(len(tokens_a), len(tokens_a) + len(tokens_b))  # Position corresponding to sentence B in input
    
    num_layers = len(query_vectors)
    for layer in range(num_layers):
        # Process queries and keys
        query_vector = query_vectors[layer][0] # assume batch_size=1; shape = [num_heads, seq_len, vector_size]
        key_vector = key_vectors[layer][0] # assume batch_size=1; shape = [num_heads, seq_len, vector_size]
        query_vectors_dict['all'].append(query_vector.tolist())
        key_vectors_dict['all'].append(key_vector.tolist())
        query_vectors_dict['a'].append(query_vector[:, slice_a, :].tolist())
        key_vectors_dict['a'].append(key_vector[:, slice_a, :].tolist())
        query_vectors_dict['b'].append(query_vector[:, slice_b, :].tolist())
        key_vectors_dict['b'].append(key_vector[:, slice_b, :].tolist())
        # Process attention
        att = atts[layer][0] # assume batch_size=1; shape = [num_heads, source_seq_len, target_seq_len]
        atts_dict['all'].append(att.tolist())
        atts_dict['aa'].append(att[:, slice_a, slice_a].tolist()) # Append A->A attention for layer, across all heads
        atts_dict['bb'].append(att[:, slice_b, slice_b].tolist()) # Append B->B attention for layer, across all heads
        atts_dict['ab'].append(att[:, slice_a, slice_b].tolist()) # Append A->B attention for layer, across all heads
        atts_dict['ba'].append(att[:, slice_b, slice_a].tolist()) # Append B->A attention for layer, across all heads

    attentions =  {
        'all': {
            'queries': query_vectors_dict['all'],
            'keys': key_vectors_dict['all'],
            'att': atts_dict['all'],
            'left_text': tokens_a + tokens_b,
            'right_text': tokens_a + tokens_b
        },
        'aa': {
            'queries': query_vectors_dict['a'],
            'keys': key_vectors_dict['a'],
            'att': atts_dict['aa'],
            'left_text': tokens_a,
            'right_text': tokens_a
        },
        'bb': {
            'queries': query_vectors_dict['b'],
            'keys': key_vectors_dict['b'],
            'att': atts_dict['bb'],
            'left_text': tokens_b,
            'right_text': tokens_b
        },
        'ab': {
            'queries': query_vectors_dict['a'],
            'keys': key_vectors_dict['b'],
            'att': atts_dict['ab'],
            'left_text': tokens_a,
            'right_text': tokens_b
        },
        'ba': {
            'queries': query_vectors_dict['b'],
            'keys': key_vectors_dict['a'],
            'att': atts_dict['ba'],
            'left_text': tokens_b,
            'right_text': tokens_a
        }
    }
    
    return attentions

def showComputation(config):
#     print("attention",config["attention"])
    att_dets = config["attention"][config["att_type"]]
    query_vector = att_dets["queries"][config["layer"]][config["att_head"]][config["query_index"]]
    keys = att_dets["keys"][config["layer"]][config["att_head"]]
    att = att_dets["att"][config["layer"]][config["att_head"]][config["query_index"]]
    
    seq_len = len(keys)
    dotProducts = []
    
    for i in range(seq_len):
        key_vector = keys[i]
        dotProduct = 0
        
        for j in range(config["vector_size"]):
            product = query_vector[j] * key_vector[j]
            dotProduct += product
        dotProducts.append(dotProduct)
    
    return dotProducts


bert_version = '/Users/bayartsogtyadamsuren/Downloads/bert-japanese-files/bert-wiki-ja'

model = BertModel.from_pretrained(bert_version)
tokenizer = BertTokenizer.from_pretrained(bert_version)

print("Head of the csv\n", f.head())
print("Number of lines", len(f))

q_x_k_scores = []
para_tokens = []
too_long = 0
errors = 0

ff.write("title\ttoken\tscore\n")

for i, x in tqdm(f.iterrows()):
    
    sentence_a = str(x["text"]).replace("\n","。").replace("〝","").replace("〞","").replace("「","").replace("」","").strip()
    sentence_b = x["title"].replace("\n","").replace("〝","").replace("〞","").replace("「","").replace("」","").strip()
    
    if len (sentence_a) > 512 or len (sentence_a) > 512:
        too_long += 1
        sentence_a = sentence_a[:512]
        sentence_b = sentence_b[:512]
    
    details_data = AttentionDetailsData(model, tokenizer)
    tokens_a, tokens_b, queries, keys, atts = details_data.get_data(sentence_a, sentence_b)
    attentions = _get_attention_details(tokens_a, tokens_b, queries, keys, atts)
    q_x_k_score = np.zeros((len(tokens_a),))

    for j, k in enumerate(tokens_b):

        config = {
            "attention": attentions,
            "att_type": "ba",
            "vector_size": 64,
            "layer": 9,
            "att_head": 6,
            "query_index": j
        }
        q_x_k_score += np.array(showComputation(config))
    
    assert len(q_x_k_score) == len(tokens_a)
    
    for j in range(len(tokens_a)):
        ff.write(f"{x['title']}\t{tokens_a[j]}\t{q_x_k_score[j]}\n")
        
    q_x_k_scores.append(q_x_k_score)
    para_tokens.append(tokens_a)
#     break
ff.close()
print("Total Too Longs: ", too_long)
print("============ Finished =============")
