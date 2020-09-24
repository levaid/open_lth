import pathlib
import os
import pandas as pd
import json
from collections import defaultdict

def parse_hparams(hparams):
    hparam_dict = defaultdict(dict)
    for line in hparams.split('\n'):
        if line[0] != ' ':
            category = line
        else:
            param, val = line.strip(' *').split('=>')
            param, val = param.strip(), val.strip()
            hparam_dict[category][param] = val
    return(hparam_dict)

def get_hparams_from_experiment(lottery_name, data_path, parameters): #['model_name', 'pruning_strategy']
    replicate = sorted(os.listdir(os.path.join(data_path, lottery_name)))[0]
    replicate_path = os.path.join(data_path, lottery_name, replicate)
    details = pathlib.Path(os.path.join(replicate_path, 'level_0/main/hparams.log')).read_text()
    levels = len(os.listdir(replicate_path))-1
    # print(details)
    details = parse_hparams(details)
    details['Pruning Hyperparameters']['pruning_levels'] = levels
    requested_params = []
    for param in parameters:
        for k, subdict in details.items():
            for subkey, value in subdict.items():
                if subkey == param:
                    requested_params += [value]
    return(requested_params)

def convert_epoch_iter(time, to, its_per_epoch=1):
    """
    Converts between iterations and epochs. 
    """
    measurement_type = time[-2:]
    if measurement_type == 'ep':
        if to == 'ep':
            return(int(time.strip('ep')))
        elif to == 'it':
            return(int(time.strip('ep')) * its_per_epoch)
        else:
            raise(NotImplementedError)
    elif measurement_type == 'it':
        if to == 'ep':
            return(float(time.strip('it')) / its_per_epoch)
        elif to == 'it':
            return(int(time.strip('it')))
        else:
            raise(NotImplementedError)
    else:
        raise(NotImplementedError)

def create_df_from_experiment(experiment_name, pruning_strat):
    df_data = []
    basepath = os.path.join('/home/levaid/bigstorage/open_lth_data', experiment_name)
    for replicate_id in sorted(os.listdir(basepath)):
        # data[replicate_id] = {}
        for level in os.listdir(os.path.join(basepath, replicate_id)):
            
            

            try:
                if level == 'level_posttrain':
                    accuracies = pd.read_csv(os.path.join(basepath, replicate_id, level, 'main', 'logger'), names = ['measure', 'unk', 'perf'])
                    top_accuracy = accuracies.query('measure == "test_accuracy"')['perf'].max()
                    df_data += [(replicate_id, level, top_accuracy, -1, pruning_strat)]
                    continue
                
                accuracies = pd.read_csv(os.path.join(basepath, replicate_id, level, 'main', 'logger'), names = ['measure', 'unk', 'perf'])
                with open(os.path.join(basepath, replicate_id, level, 'main', 'sparsity_report.json')) as f:
                    pruning_report = json.load(f)
                top_accuracy = accuracies.query('measure == "test_accuracy"')['perf'].max()
                pruning_percent = pruning_report['unpruned'] / pruning_report['total']
                #data[replicate_id][level] = {'accuracy': top_accuracy, 'unpruned': pruning_percent}
                # data[replicate_id][level] = top_accuracy, pruning_percent
                df_data += [(replicate_id, level, top_accuracy, pruning_percent, pruning_strat)]

            except Exception as e:
                print(replicate_id, level, e)
                
            
                
    df = pd.DataFrame(df_data, columns = ['replicate_id', 'level', 'accuracy', 'unpruned', 'pruning_strategy'])
    return(df)


     