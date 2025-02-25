# A Taxonomy of Urban Façade Defects and Their Distribution on Façade Components: A Data-driven Analysis of Historical Inspection Reports

## Overview
* This repository contains code for two key tasks using Natural Language Processing (NLP):
* Façade Components & Defects Tagging: Automatically tag urban façade defects and their distribution on façade components in inspection reports.
* Clustering: Group building reports (using GEOID or building code as the key) based on various defect features extracted from the reports.

## Façade Components & Defects Tagging
* The following snippet shows how to prepare the data, build a BiLSTM-CRF model, train it, and test it on sample sentences.
```
# Prepare the data
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

src_tokenizer = Tokenizer(oov_token='OOV')
src_tokenizer.fit_on_texts(sentences)
X_data = src_tokenizer.texts_to_sequences(sentences)
X_data = pad_sequences(X_data, padding='post', maxlen=max_len)

# Training
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional
from keras_contrib.layers import CRF

model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(TimeDistributed(Dense(50, activation='relu')))
crf = CRF(tag_size)
model.add(crf)
model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy])
history = model.fit(X_train, y_train, batch_size=32, epochs=1, validation_split=0.2,
                    callbacks=[F1score(use_char=False)])

# Testing
# Manual prediction on a test sentence
new_sentence = X_test_[5208]
new_encoded = [word_to_index.get(w, 1) for w in new_sentence]
new_padded = pad_sequences([new_encoded], padding="post", maxlen=max_len)
p = bilstm_crf_model.predict(np.array([new_padded[0]]))
p = np.argmax(p, axis=-1)

print("{:15}||{}".format("word", "pred"))
for w, pred in zip(new_sentence, p[0]):
    print("{:15}: {:5}".format(w, index_to_ner[pred]))
```

## Clustering
* The clustering code analyzes façade inspection reports by grouping building codes based on features extracted from the defect descriptions. For example, the following defect features are used:
* Structural Deformation: displacement, deflection, slippage, bowing/bulging
* Repair Failures & Connection Issues: repair failure, loose, fastener failure, failure, coating failure/peeling/flaking, separation
* Cracking & Damage: crack, erosion/pitting, abrasion, chipping, scaling, spall, delamination, cavity

![image](image)

* The main working code section for clustering is shown below:
```
# clustering code
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# [after found optimal number of cluster]

# Final clustering using the optimal cluster count (opt_clust)
kmeans_final = KMeans(n_clusters=opt_clust, random_state=0).fit(clustDefe1)
labels_final = kmeans_final.predict(clustDefe1)

# Create a DataFrame that links each GEOID to its cluster label
res1 = pd.DataFrame({'GEOID': clustDefe1.index, 'label': labels_final})

# visuliazation

import matplotlib.pyplot as plt

# Merge spatial data with clustering results and plot the clusters
ax = nyct2020_MA.merge(res1, on='GEOID', how='left').plot(
    column='label', legend=True, edgecolor='black',
    legend_kwds={'loc': 'lower right'},
    missing_kwds={
        "color": "lightgrey",
        "edgecolor": "black",
        "linewidth": 2,
        "alpha": 0.5,
        "hatch": "///",
        "label": "Other Clusters"
    }
)
plt.show()
```
