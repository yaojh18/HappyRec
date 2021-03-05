# coding=utf-8

from happyrec.utilities.commandline import *
from happyrec.tasks.Task import Task


# from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
# # # ignore 'BrokenPipeError: [Errno 32] Broken pipe'
# signal(SIGPIPE, SIG_IGN)


def main():
    args, task_args, model_args, trainer_args = parse_cmd_args(task_name='Task')
    task = Task(**vars(task_args), model_args=vars(model_args), trainer_args=vars(trainer_args))
    task.run()
    return


if __name__ == '__main__':
    main()
