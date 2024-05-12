import argparse


def get_args():
    """
    Description:
    Parses arguments at command line.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode', dest='mode', type=str, default='train')
    parser.add_argument('-c', '--config', dest='config', type=str)
    parser.add_argument('-e', '--env', dest='env', type=str, default='PandaReach-v3')
    parser.add_argument( '--model', dest='model', type=str, default='DDPG')

    args = parser.parse_args()

    return args
