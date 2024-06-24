import skdim
import pickle
import sys

states_pickle_string = sys.argv[1]
states_pickle_count = int(sys.argv[2])
id_method = sys.argv[3]
out_pickle_prefix = sys.argv[4]

states = dict()
for index in range(states_pickle_count):
    pickle_i = index + 1
    states_pickle =  str(pickle_i) + "_" + states_pickle_string + ".pickle"
    with open(states_pickle,'rb') as f:
        curr_states = pickle.load(f)
    for i in curr_states:
        if not i in states:
            states[i] = []
        states[i] = states[i] + curr_states[i]

id_estimates = dict()

for layer in states:
    print("processing layer " + str(layer),file=sys.stderr)
    if id_method == "PCA":
        id_estimates[layer] = skdim.id.lPCA().fit_transform_pw(states[layer],n_neighbors=100)    
    else:
        id_estimates[layer] = skdim.id.MLE().fit_transform_pw(states[layer],n_neighbors=100)    

out_pickle_name = out_pickle_prefix + ".pickle"

with open(out_pickle_name, 'wb') as f:
    pickle.dump(id_estimates, f)


# python multi_input_get_by_layer_ids.py bloom-3b_en-es_english_europarl_residual 2 MLE bloom-3b_en-es_english_europarl_MLE_id

