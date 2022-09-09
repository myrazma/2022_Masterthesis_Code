"""
Module for output formatter to upload predictions to codalab EMP competiton    
"""

from ast import arg
import sys
import os
import pandas as pd


def generate_result_file(model_name, output_dir='../output/', store_as_tsv=True, columns=['empathy', 'distress'], output_file_name='predictions_EMP', split='test'):
    """Loading from results file of a model and generate an output file

    Args:
        model_name (_type_): _description_
        output_dir (str, optional): _description_. Defaults to '../output/'.
        store_as_tsv (bool, optional): _description_. Defaults to True.
        columns (list, optional): _description_. Defaults to ['empathy', 'distress'].
    """
    model_dir = output_dir + model_name + '/'

    subdirs = [x[0] for x in os.walk(model_dir)] 
    subdirs = [subdir for subdir in subdirs if subdir[-1]!='/']
    # check which subtasks_are available
    output_task_paths = {subdir.split('/')[-1]: subdir for subdir in subdirs}

    test_result_file_name = f'{split}_results'
    results_df = pd.DataFrame()
    for col in columns:
        if col in output_task_paths.keys():
            test_file_path = output_task_paths[col] + f'/{test_result_file_name}_{col}.txt'
            task_results = pd.read_csv(test_file_path, sep='\t', index_col=0, header=0)['prediction']
            results_df[col] = task_results
        else:
            print(f'For {model_name}: Task {col} not available. Will not be stored')

    # sort by columns: make sure, empathy is the first column. Important for the tsv generation
    results_df = results_df.sort_index(axis=1, ascending=False)

    if store_as_tsv:
        results_df.to_csv(model_dir + output_file_name + '.tsv', index=None, header=None, sep='\t')
    else:
        results_df.to_csv(model_dir + output_file_name + '.csv')


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print('No input available. Please input a model name.')
        sys.exit(-1)
    
    args = sys.argv[1:]  # ignore first argument (name of file)
    split_arg = 'test'  # as base if not other defined
    if '--split' in args:
        split_arg_idx = int(args.index('--split') + 1)
        if split_arg_idx < len(args):
            split_arg = args[split_arg_idx]
        args.pop(split_arg_idx - 1)
        args.pop(split_arg_idx -1)

    if split_arg == 'test':
        output_file_name='predictions_EMP'
    else:
        output_file_name=f'predictions_EMP_{split_arg}'


    for model_name in args:
        try:
            generate_result_file(model_name, store_as_tsv=True, split=split_arg, output_file_name=output_file_name)
        except Exception as e:
            print(f"MyWarning: Wasn't able to generate output from the following model {model_name}. Exception occured:\n {e}")

