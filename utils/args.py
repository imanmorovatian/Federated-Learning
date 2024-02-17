import argparse


def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--dataset', type=str, choices=['idda', 'femnist', 'rfemnist'], required=True, help='dataset name')

    parser.add_argument('--niid', action='store_true', default=False,
                        help='Run the experiment with the non-IID partition (IID by default). Only on FEMNIST dataset.')
    
    parser.add_argument('--model', type=str, choices=['deeplabv3_mobilenetv2', 'resnet18', 'cnn'], required=True,
                        help='model name')
    
    parser.add_argument('--num_rounds', type=int, required=True, help='number of rounds')
    parser.add_argument('--num_epochs', type=int, required=True, help='number of local epochs')
    parser.add_argument('--clients_per_round', type=int, required=True, help='number of clients trained per round')

    parser.add_argument('--client_selection_m', type=str, choices=['random', 'pwc'], required=True,
                        help='The method for client selection. It could be random or power of choice (pwc)')
    
    parser.add_argument('--num_candidates', type=int, default=10, help='number of candidates for power-of-choice strategy')
    parser.add_argument('--hnm', action='store_true', default=False, help='Use hard negative mining reduction or not')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--bs', type=int, default=4, help='batch size')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--m', type=float, default=0.9, help='momentum')

    parser.add_argument('--dgm', type=str, choices=['none', 'fedsr', 'feddg'], required=True,
                        help='The used method to tackle domain generalization')
    
    parser.add_argument('--dt', type=int, choices=[0, 1, 2, 3, 4, 5], default=0,
                        help='The domain will be used at the domain test')
    
    parser.add_argument('--fedsr_l2r', type=float, default=0.01, help='L2R coefficient')
    parser.add_argument('--fedsr_cmi', type=float, default=0.001, help='Conditional Mutual Information')
    
    return parser
