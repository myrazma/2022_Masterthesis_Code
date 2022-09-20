# Data

## Data Tree Structure 

The tree structure of the data files and subfolders. If all data sets are downloaded it should look like this:

```
data/
├── buechel_empathy/                        (Buechel et al., 2018)
│   ├── articles_adobe_AMT.csv
│   ├── goldstandard_dev_2022.tsv
│   ├── messages_dev_features_ready_for_WS_2022.tsv
│   ├── messages_dev_sentencized_automatic_emotion_tags.tsv
│   ├── messages_train_ready_for_WS.tsv
│   └── messages_train_sentencized_automatic_emotion_tags.tsv
├── lexicon/
│   ├── distress/                           (Sedoc et al., 2019)
│   │   ├── distress_clusters.txt
│   │   └── distress_lexicon.txt
│   ├── empathy/                            (Sedoc et al., 2019)
│   │   ├── empathy_clusters.txt
│   │   └── empathy_lexicon.txt
│   ├── NRC-Emotion-Intensity-Lexicon-v1/   (Mohammad et al., 2018)
│   │   ├── NRC-Emotion-Intensity-Lexicon-v1.txt
│   │   └── ...
│   ├── NRC-VAD-Lexicon-Aug2018Release/     (Mohammad et al., 2018)
│   │   ├── NRC-VAD-Lexicon.txt
│   │   └── ...
│   └── BRM-emot-submit.csv                 (Warriner et al., 2013)
├── MoRT_projection
│   └── projection_model.p                  (Schramowski et al., 2021)
└── trained_adapters
    ├── bert-base-uncased_pf_emotion         (Poth et al., 2020)  
    ├── bert-base-uncased-pf-social_i_qa         (Poth et al., 2020)  
    ├── distress          
    ├── empathy                  
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


## MoRT: Projection onto the moral dimension

This is a pickle file from the moral projection containing the generated projection onto the moral dimension. This file contains a .. dimensional vector.

Schramowski et al. 2021 - Large Pre-trained Language Models Contain Human-like Biases of What is Right and Wrong to Do: https://arxiv.org/abs/2103.11790

The data can be downloaded from their GitHub repository: https://github.com/ml-research/MoRT_NMI/tree/master/MoRT/data/subspace_proj/bert-large-nli-mean-tokens

# Source
Buechel, S., Buffone, A., Slaff, B., Ungar, L., & Sedoc, J. (2018). Modeling empathy and distress in reaction to news stories. arXiv preprint arXiv:1808.10399. https:
//arxiv.org/abs/1808.10399

Mohammad, S. (2018). Obtaining reliable human ratings of valence, arousal, and dom- inance for 20,000 English words. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 174–184. https: //doi.org/10.18653/v1/P18-1017

Poth, C., Pfeiffer, J., Rücklé, A., & Gurevych, I. (2021). What to pre-train on? Efficient in- termediate task selection. Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 10585–10605. https://doi.org/10.18653/v1/ 2021.emnlp-main.827

Sedoc, J., Buechel, S., Nachmany, Y., Buffone, A., & Ungar, L. (2020). Learning word ratings for empathy and distress from document-level user responses. Proceedings of the 12th Language Resources and Evaluation Conference, 1664–1673. https: //aclanthology.org/2020.lrec-1.206/

Warriner, A. B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 english lemmas. Behavior Research Methods, 45(4), 1191– 1207. https://doi.org/10.3758/s13428-012-0314-x