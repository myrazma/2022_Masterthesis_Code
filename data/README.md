# Data

Overview of the structure for the data files and subfolders (if all data sets are downloaded):

```
data/
├── buechel_empathy/
│   ├── articles_adobe_AMT.csv
│   ├── goldstandard_dev_2022.tsv
│   ├── messages_dev_features_ready_for_WS_2022.tsv
│   ├── messages_dev_sentencized_automatic_emotion_tags.tsv
│   ├── messages_train_ready_for_WS.tsv
│   └── messages_train_sentencized_automatic_emotion_tags.tsv
├── lexicon/
│   ├── distress/
│   │   ├── distress_clusters.txt
│   │   └── distress_lexicon.txt
│   ├── empathy/
│   │   ├── empathy_clusters.txt
│   │   └── empathy_lexicon.txt
│   ├── NRC-Emotion-Intensity-Lexicon-v1/
│   │   └── ...
│   ├── NRC-VAD-Lexicon-Aug2018Release/
│   │   └── ...
│   └── BRM-emot-submit.csv
```

## Empathy Data set
The data can be downloaded from the Codalab Competition: https://codalab.lisn.upsaclay.fr/competitions/834#learn_the_details-datasets

#### Example
| message_id          | response_id       | article_id | empathy           | distress | empathy_bin | distress_bin | essay              |
|---------------------|-------------------|------------|-------------------|----------|-------------|--------------|--------------------|
| R_1hGrPtWM4SumG0U_1 | R_1hGrPtWM4SumG0U | 67         | 5.667 | 4.375    | 1           | 1            | it is really ..    |
| R_1hGrPtWM4SumG0U_2 | R_1hGrPtWM4SumG0U | 86         | 4.833             | 4.875    | 1           | 1            | the phone lines .. |

## Lexical Data

The lexical data can be downloaded at the following links and **added into the subfolder** *lexicon* as they are.



### Lexicon for empathetic concern and personal distress
The lexicon dataset for word rating for empathic concern and personal distress can be downloaded here: http://www.wwbp.org/lexica.html

#### Example
| word    | rating |
|----------|------------------|
| helps    | 4.31595359173709 |
| uncommon | 2.53496437756067 |
| blank    | 3.55986254538211 |
