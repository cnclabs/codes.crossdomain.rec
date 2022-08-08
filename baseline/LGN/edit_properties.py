import configparser
import argparse

config_name = "CPR_LightGCN.properties"

parser=argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, help='name of dataset', default=None)
parser.add_argument('--gpu', type=int, help='gpu id', default=None)
args=parser.parse_args()
#assert args.dataset is not None, "Please enter the dataset name as an argument"

config = configparser.ConfigParser()
config.read(config_name)

if args.dataset is not None:
    config['default']['data.input.dataset'] = args.dataset
if args.gpu is not None:
    config['default']['gpu_id'] = args.gpu

with open(config_name, 'w') as f:
    config.write(f)