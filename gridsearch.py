# coding=utf-8

import argparse
import sys

from happyrec.utilities.commandline import *
from happyrec.tasks.GridSearch import GridSearch


def main():
    grid_parser = argparse.ArgumentParser(description='GridSearch Args')
    grid_parser = GridSearch.add_task_args(grid_parser)
    grid_args, _ = grid_parser.parse_known_args()
    gridsearch = GridSearch(**vars(grid_args))
    gridsearch.run()
    return


if __name__ == '__main__':
    main()
