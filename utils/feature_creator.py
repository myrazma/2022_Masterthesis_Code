
import pandas as pd
import numpy as np
import pickle

# own modules
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))


import utils.utils as utils
from EmpDim.pca import create_pca
from utils.arguments import PCAArguments, DataTrainingArguments
from EmpDim.funcs_mcm import BERTSentence


class FeatureCreator():
    def __init__(self, pca_args=None, data_args=None, device='cpu'):
        self.device = device
        self.data_root_folder = data_args.data_dir
        self.empathy_lex, self.distress_lex = utils.load_empathy_distress_lexicon(self.data_root_folder)
        self.lexicon_dict = {'empathy': self.empathy_lex, 'distress': self.distress_lex}  # lexicon where we can get the features by key / task_name
        self.articles = None

        self.__pca_dict = {}
        self.pca_args = PCAArguments if pca_args is None else pca_args
        self.data_args = DataTrainingArguments if data_args is None else data_args
    
    def create_lexical_feature_dataframe(self, data_pd, column_name='essay_tok', task_name=['empathy', 'distress']):
        # create lexical feature for a pandas dataframe
        def emp_dis_calc_exp_word_rating_score(lexicon, essay):
            # using exponential function of ratings -> higher values will have more importance
            rating_scores = []
            for word in essay:
                rating_scores.append(np.exp(lexicon[word]))

            rating_scores_np = np.array(rating_scores)
            average_rating = sum(rating_scores_np) / len(rating_scores_np)
            return average_rating
        
        if not isinstance(task_name, list):
            task_name = [task_name]
        
        for task in task_name:
            if not task in self.lexicon_dict.keys():
                print(f'MyWarning: The task {task} is not known. No features will be created for it.')
                continue
            col_name = f'{task}_word_rating'
            data_pd[col_name] = data_pd[column_name].apply(lambda x: emp_dis_calc_exp_word_rating_score(self.lexicon_dict[task], x))

        return data_pd

    def create_lexical_feature(self, essays_tok, task_name):
        """_summary_

        Args:
            essays_tok (list(str)): The tokenized essays
            task_name (str or list(str)): The

        Returns:
            _type_: _description_
        """
        # essays_tok should be tokenized essay
        def emp_dis_calc_exp_word_rating_score(lexicon, essay):
            # using exponential function of ratings -> higher values will have more importance
            rating_scores = []
            for word in essay:
                rating_scores.append(np.exp(lexicon[word]))

            rating_scores_np = np.array(rating_scores)
            average_rating = sum(rating_scores_np) / len(rating_scores_np)
            return average_rating
        
        results = []

        if not isinstance(task_name, list):
            task_name = [task_name]
        for task in task_name:
            if not task in self.lexicon_dict.keys():
                print(f'MyWarning: The task {task} is not known. No features will be created for it.')
                continue
            essays_ratings = np.array(list(map(lambda x: emp_dis_calc_exp_word_rating_score(self.lexicon_dict[task], x), essays_tok)))
            results.append(essays_ratings)
        if len(results) == 0:
            print('MyWarning: create_lexical_feature() - Results Empty')
            return []
        return results if len(results) > 1 else results[0]

    def get_pca(self, task):
        """Create PCA from the class PCA dim and fit it to the lexical data
        Here: Only get the pca, the actual fitting / loading or whatever should be in the pca.py script

        task can either be distress or empathy
        """
        if task not in self.__pca_dict.keys():
            task_lexicon = self.lexicon_dict[task]  # TODO: I am not using the lexicon here, to we have to?
            # TODO get all of this information:
            dim_pca = create_pca(my_args=self.pca_args, data_args=self.data_args, tensorboard_writer=None, device=self.device)
            self.__pca_dict[task] = dim_pca

        return self.__pca_dict[task]

    def get_articles(self):
        if self.articles == None:
            self.articles = utils.load_articles(data_root_folder=self.data_args.data_dir)
            
        return self.articles

    def create_pca_feature(self, essays, task_name):
        results = []  # len(results) == len(task_name) if all items in task_name are valid tasks
        if not isinstance(task_name, list):
            task_name = [task_name]

        for task in task_name:
            if not task in self.lexicon_dict.keys():
                print(f'MyWarning: The task {task} is not known. No features will be created for it.')
                continue

            # --- do pca ---
            if task not in self.__pca_dict.keys():  # if the PCA for this specific task is not in dict, get it!
                self.get_pca(task)
            task_pca = self.__pca_dict[task]  # get the pca object

            # Transform essays into sentence model embeddings
            essay_embeddings = task_pca.sent_model.get_sen_embedding(essays)
            # Transform the embeddings of the essays with pca
            essays_emp_dim = task_pca.transform(essay_embeddings)

            results.append(essays_emp_dim)  # append the features for the essays to results list

        if len(results) == 0:
            print('MyWarning: create_pca_feature() - Results Empty')
            return []
        return results if len(results) > 1 else results[0]
    
    def load_mort_pca(self, data_dir='../data', filename='/MoRT_projection/projection_model.p'):
        file = open(data_dir + filename, 'rb')
        # dump information to that file
        data = pickle.load(file)
        # close the file
        file.close()

        mort_pca = None
        try: 
            mort_pca = data['projection']
        except Exception as e:
            print(f'\n No pca for MoRT found. PCA object will be None. Error: \n{e}')
        return mort_pca  # if not found, than pca will be None

    def create_MoRT_feature(self, essays, principle_components_idx=None):
        """_summary_

        Args:
            essays (_type_): _description_
            principle_components (list(int), optional): A list of indices of the principle components ot select. 
                                                    Only accepts input that are actualy idx for pcs in this pca (up until idx 4). 
                                                    Defaults to None.
        """
        sent_model = BERTSentence(device=self.device)
        essay_embeddings = sent_model.get_sen_embedding(essays)

        try:
            mort_pca = self.load_mort_pca(data_dir=self.data_args.data_dir)
        except Exception as e:
            print(f'\n Could not load MoRT PCA. Error: \n{e}')
            sys.exit(-1)

        moral_dim = mort_pca.transform(essay_embeddings)
        mort_dimension = moral_dim.shape[1]
        if principle_components_idx is not None and isinstance(principle_components_idx, list):  # select principle components
            # filter out idx that are not valid
            principle_components_idx = [idx for idx in principle_components_idx if (idx <= mort_dimension-1) and (idx >= 0)]
            moral_dim = moral_dim[:, principle_components_idx]

        return moral_dim

    def create_MoRT_feature_articles(self, essay_article_ids, principle_components_idx=None):
        """Create MoRT Features of the articles to the corresponding essays

        Args:
            essay_article_ids (_type_): The arictle ids of the different essays 
                                        (The sorting of this is crucial: Should be the same as the training data)
            principle_components_idx (_type_, optional): _description_. Defaults to None.
        """
        # get the articles
        try:
            articles = self.get_articles()
        except Exception as e:
            print(f'Could not load articles:\n {e}')
            return None
        articles_text = articles['text']
        article_ids = articles['article_id']

        # transform the articles into featurespace
        moral_dim_articles = self.create_MoRT_feature(self, articles_text, principle_components_idx=principle_components_idx)

        # map articles on essay article ids: result should have the same length as essays bzw essay_article_ids
        # moral_dim_articles: x * y
        # article_ids: x
        # essay_article_ids: z
        # output: z * y
        print(moral_dim_articles.shape)
        print(article_ids.shape)
        print(essay_article_ids.shape)
        article_ids_list = list(article_ids)
        indices = [article_ids_list.index(id) for id in list(essay_article_ids)]
        article_mort_per_essay = np.take(moral_dim_articles, indices, axis=0)
        try:
            print('article_mort_per_essay', article_mort_per_essay)
        except:
            print('No article_mort_per_essay')
        
        try:
            print('article_mort_per_essay.shape', article_mort_per_essay.shape)
        except:
            print('No article_mort_per_essay.shape')

        try:
            print('article_mort_per_essay.size()', article_mort_per_essay.size())
        except:
            print('No article_mort_per_essay.size()')

        return article_mort_per_essay










    
        

