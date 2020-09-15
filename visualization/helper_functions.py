import pathlib
import os
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


     