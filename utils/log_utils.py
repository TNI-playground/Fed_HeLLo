import os
import datetime

def set_log_path(args):
    '''
    log path for different datasets and methods
    '''
    path =  './log/' + args.dataset +'/' + args.model + '/' + args.method + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    path_log = os.path.join(path, args.config_name.split('.')[0])
    
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    everything_record_path = path_log + '_' + str(timestamp)
    if not os.path.exists(everything_record_path):
        os.makedirs(everything_record_path)
    return everything_record_path