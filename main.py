import argparse
from model import Model
from utils import *


def parse_args():
    desc = "Pytorch implementation of MakeItTalk"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')
    parser.add_argument('--dataset', type=str, default='dataset', help='dataset_name')

    parser.add_argument('--iteration', type=int, default=1000000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=100, help='The number of print freq')
    parser.add_argument('--save_freq', type=int, default=100000, help='The number of model save freq')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.000001, help='The weight decay')
    parser.add_argument('--lambda_c', type=float, default=1., help='Weight for content loss')
    parser.add_argument('--lambda_s', type=float, default=1., help='Weight for speaker aware loss')
    parser.add_argument('--mu_s', type=float, default=0.001, help='Weight for speaker aware loss')

    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the results')
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--use_mouth_weight', type=str2bool, default=False)
    parser.add_argument('--use_motion_loss', type=str2bool, default=False)

    return check_args(parser.parse_args())


def check_args(args):
    # --result_dir
    check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'test'))

    # --phase
    assert args.phase in ['train', 'test']

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # open session
    model = Model(args)

    # build graph
    model.build_model()

    if args.phase == 'test':
        model.test()
        print(" [*] Test finished!")
    else:
        model.train()
        print(" [*] Training finished!")


if __name__ == '__main__':
    main()
