import argparse

class Config(object):
    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)


def get_config(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser("configuration setting")
    parser.add_argument('--batch_size', default=128, type=int, help='the number of batch size')
    parser.add_argument('--model_type', required=True, default='ConTL')
    parser.add_argument('--data-path', help='path to the data(SEED4, DEAP)')
    parser.add_argument('--data-choice', required=True, help='4: SEED-IV, deap: DEAP')
    parser.add_argument('--learning_rate', type=float, default= 0.0002)
    parser.add_argument('--w-mode', default= 'w')
    parser.add_argument('--num_epochs',type=int, default=500)
    parser.add_argument('--b1', default=0.5)
    parser.add_argument('--b2', default=0.999)
    parser.add_argument('--n-classes', required=True, type=int, help='number of classes')
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--label_type', default='valence_labels')
    parser.add_argument('--lstm_hidden_size', type=int, default=8)
    parser.add_argument('--subject', type=str, default='1')
    parser.add_argument('--lstm', action='store_true')
    parser.add_argument('--save_file_name', required=True)
    parser.add_argument('--n_units', type=int, default=106)
    
    if parse:
        args=parser.parse_args()


    if args.data_choice=='4':
        args.n_classes=4
    elif args.data_choice=='deap':
        args.n_classes=2
    else:
        print('No dataset mentioned')
        exit()
    
    args = vars(args)
    args.update(optional_kwargs)

    return Config(**args)














#
