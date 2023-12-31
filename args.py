"""
Command-line arguments for train and test model.
"""

import argparse


def get_classifier_args():
    """Get arguments for train_classifier.py
    """
    parser = argparse.ArgumentParser('Train the model')
    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--lr',
                        type=float,
                        default=0.0005,
                        help='Learning rate.')

    parser.add_argument('--l2_wd',
                        type=float,
                        default=3e-7,
                        help='L2 weight decay.')

    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')

    parser.add_argument('--eval_steps',
                        type=int,
                        default=30000,
                        help='Number of steps between successive evaluations.')

    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs for which to train. Negative means forever.')

    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')

    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')

    parser.add_argument("--top_gene",
                        type=int,
                        default=8192,
                        help="Top gene to select in the analysis")

    parser.add_argument("--alpha",
                        type=float,
                        default=1.5,
                        help="alpha when computing path weight")

    parser.add_argument("--reg_weight",
                        type=float,
                        default=0.1,
                        help="weight on the regularization term")

    parser.add_argument("--reg_mode",
                        type=str,
                        default="deg",
                        choices=("deg", "up", "down"),
                        help="Regularization mode for selecting different type of paths")

    args = parser.parse_args()

    if args.metric_name == 'MSE':
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name in ('Accuracy', 'F1', "AUC"):
        # Best checkpoint is the one that maximizes EM or F1
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')
    return args


def add_train_test_args(parser):
    """Add arguments common to train and test files"""
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify training or test run.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=0,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.1,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--output_size',
                        type=int,
                        default=2,
                        help='Output size of the model(number of classes).')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')
    parser.add_argument('--JK',
                        type=str,
                        default="max",
                        choices=("concat", "last", "max", "sum"),
                        help="Jump knowledge method"
                        )

    parser.add_argument('--input_size',
                        type=int,
                        default=1,
                        help='Number of features in input data.')

    parser.add_argument('--hidden_size',
                        type=int,
                        default=128,
                        help='Number of features in encoder hidden layers.')

    parser.add_argument('--r',
                        type=int,
                        default=8,
                        help='Number of head in path aggregation.')

    parser.add_argument('--gamma',
                        type=int,
                        default=2,
                        help='Gamma in high order softmax.')

    parser.add_argument('--metric_name',
                        type=str,
                        default='AUC',
                        choices=('MSE', "Accuracy", "F1", "AUC"),
                        help='Name of dev metric to determine best checkpoint.')

    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility.')

    parser.add_argument('--head',
                        type=int,
                        default=8,
                        help='Head number in PathFormer')

    parser.add_argument('--num_layer',
                        type=int,
                        default=6,
                        help='Number of layer in PathFormer')

    parser.add_argument('--num_edges',
                        type=int,
                        default=6,
                        help='Number of edge type in the dataset')

    parser.add_argument('--max_length',
                        type=int,
                        default=10,
                        help='Maximum number of length for each sampled path.')

    parser.add_argument('--add_coexp',
                        action="store_true",
                        help='add co_expression gene interaction into network database during training and testing.')

    parser.add_argument('--val_ratio',
                        type=float,
                        default=0.1,
                        help='Validation dataset ratio.')

    parser.add_argument('--test_ratio',
                        type=float,
                        default=0.2,
                        help='Test dataset ratio.')


    parser.add_argument('--runs',
                        type=int,
                        default=5,
                        help='Number of repeated runs.')



def add_common_args(parser):
    """Add arguments common to all scripts"""
    parser.add_argument('--network_database_path',
                        type=str,
                        default="data/network/processed_network.npz",
                        help="Path of network database")
    parser.add_argument('--gs_path',
                        type=str,
                        default="data/gene_feature/gene_gs_dict.json",
                        help="Path of gene set feature")

    parser.add_argument('--gene_symbol_file_path',
                        type=str,
                        default="data/ad_mice/gene_list.npz",
                        help="Gene symbol list for the input data.")

    parser.add_argument('--control_file_path',
                        type=str,
                        default="data/ad_mice/TAFE4_tam_ex.txt",
                        help="Path of gene set feature")

    parser.add_argument('--test_file_path',
                        type=str,
                        default="data/ad_mice/TAFE4_oil_ex.txt",
                        help="Path of gene set feature")
