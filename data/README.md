# Data

## Data Tree Structure 

The tree structure of the data files and subfolders. If all data sets are downloaded it should look like this:

```
data/
├── buechel_empathy/                        (Buechel2018)
│   ├── articles_adobe_AMT.csv
│   ├── goldstandard_dev_2022.tsv
│   ├── messages_dev_features_ready_for_WS_2022.tsv
│   ├── messages_dev_sentencized_automatic_emotion_tags.tsv
│   ├── messages_train_ready_for_WS.tsv
│   └── messages_train_sentencized_automatic_emotion_tags.tsv
└── lexicon/
    ├── distress/                           (Sedoc2019)
    │   ├── distress_clusters.txt
    │   └── distress_lexicon.txt
    ├── empathy/                            (Sedoc2019)
    │   ├── empathy_clusters.txt
    │   └── empathy_lexicon.txt
    ├── NRC-Emotion-Intensity-Lexicon-v1/   (Mohammad2018)
    │   ├── NRC-Emotion-Intensity-Lexicon-v1.txt
    │   └── ...
    ├── NRC-VAD-Lexicon-Aug2018Release/     (Mohammad2018b)
    │   ├── NRC-VAD-Lexicon.txt
    │   └── ...
    └── BRM-emot-submit.csv                 (Warriner2013)
```

## Empathy Data set

Buechel et al., 2018 - Modeling empathy and distress in reaction to news stories: https://arxiv.org/abs/1808.10399

The data can be downloaded from the Codalab Competition: https://codalab.lisn.upsaclay.fr/competitions/834#learn_the_details-datasets

After downloading store the data in the subfolder */data/buechel_empathy/* as seen in the [Data Tree Structure](#Data-Tree-Structure).

#### Example
| message_id          | response_id       | article_id | empathy           | distress | empathy_bin | distress_bin | essay              |
|---------------------|-------------------|------------|-------------------|----------|-------------|--------------|--------------------|
| R_1hGrPtWM4SumG0U_1 | R_1hGrPtWM4SumG0U | 67         | 5.667 | 4.375    | 1           | 1            | it is really ..    |
| R_1hGrPtWM4SumG0U_2 | R_1hGrPtWM4SumG0U | 86         | 4.833             | 4.875    | 1           | 1            | the phone lines .. |

## Lexical Data


The lexical data can be downloaded at the following links and should be added into the subfolder */data/lexicon/* as they are (keep the folder struture as seen in the [Data Tree Structure](#Data-Tree-Structure)).

### Lexicon for empathetic concern and personal distress

Sedoc et al., 2019 - Learning word ratings for empathy and distress from document-level user responses: http://www.wwbp.org/lexica.html

The lexicon dataset for word rating for empathic concern and personal distress can be downloaded here: http://www.wwbp.org/lexica.html

#### Example
| word    | rating |
|----------|------------------|
| helps    | 4.31595359173709 |
| uncommon | 2.53496437756067 |
| blank    | 3.55986254538211 |


### NRC - Affective ratings: Emotion intensity

Mohammad, 2018 - Word Affect Intensities: https://aclanthology.org/L18-1027

The data can be downloaded here: http://saifmohammad.com/WebPages/AffectIntensity.htm . The downloaded folder structure can be kept, just add the folder into */data/lexicon/* as seen in the [Data Tree Structure](#Data-Tree-Structure).

It comes in various different languages, the main language is english. The file for the english version is *NRC-Emotion-Intensity-Lexicon-v1.txt*, we only need this file for now.

The lexicon constist of 5,961 english words with associated emotions and their emotion-intensity-score. A word can be associated with mulitple emotions, but does not have to be associated wiht all emotions.

##### Example
| word          | 	emotion	| emotion-intensity-score   |
|---------------|-----------|---------------------------|
| kill          | fear      |  0.962                    |
| kill          | sadness   |  0.797                    |
| respect       | joy       |  0.500                    |
| wonderfully   | joy       | 0.844                     | 


### NRC - Valence, arousal and dominance

Mohammad, 2018 (b) - Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words: https://aclanthology.org/P18-1017

The data can be downloaded here: http://saifmohammad.com/WebPages/nrc-vad.html . The downloaded folder structure can be kept, just add the folder into */data/lexicon/* as seen in the [Data Tree Structure](#Data-Tree-Structure).

It comes in various different languages, the main language is english. The file for the english version is *NRC-VAD-Lexicon.txt*, we only need this file for now.


For around 20,000 english words, this data set contains a rating for valence, arousal and domninance.

| Word          | Valence	| Arousal   | Dominance | 
|---------------|-----------|-----------------------|
| achieve       |   0.816   |  0.545    |	0.843   |    
| uncherished   |   0.250	|  0.412    |	0.280   |
| sue           |	0.219	|  0.731	|   0.679   |

##### Example

Warriner et al., 2013 - Norms of valence, arousal, and dominance for 13,915 English lemmas: https://doi.org/10.3758/s13428-012-0314-x

The data can be downloaded here: https://link.springer.com/article/10.3758/s13428-012-0314-x#Abs1

### Valence, arousal and dominance ratings
