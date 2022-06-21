
import pandas as pd
import numpy as np

# own modules
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))


import utils.utils as utils
from EmpDim.pca import create_pca
from utils.arguments import PCAArguments, DataTrainingArguments


class FeatureCreator():
    def __init__(self, pca_args=None, data_args=None, device='cpu'):
        self.device = device
        self.data_root_folder = data_args.data_dir
        self.empathy_lex, self.distress_lex = utils.load_empathy_distress_lexicon(self.data_root_folder)
        self.lexicon_dict = {'empathy': self.empathy_lex, 'distress': self.distress_lex}  # lexicon where we can get the features by key / task_name

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
        

