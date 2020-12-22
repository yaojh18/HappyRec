# coding=utf-8

from happyrec.utilities.argument import *
from happyrec.tasks.Task import Task


def main():
    args, task_args, model_args, trainer_args = parse_cmd_args()
    Task().run(args=args, task_args=task_args, model_args=model_args, trainer_args=trainer_args)
    return


if __name__ == '__main__':
    main()
