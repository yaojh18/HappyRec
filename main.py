# coding=utf-8

import socket
from happyrec.utilities.commandline import *
from happyrec.tasks.Task import Task


# from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
# # # ignore 'BrokenPipeError: [Errno 32] Broken pipe'
# signal(SIGPIPE, SIG_IGN)

def correct_cuda_no(cuda_str):
    cudas = cuda_str.strip().split(',')
    cudas = [c.strip() for c in cudas]
    hostname = socket.gethostname()
    if hostname == 'deepest':
        correct_dict = {'1': '3',
                        '2': '4',
                        '3': '1',
                        '4': '2'}
    else:
        correct_dict = {}
    cudas = [correct_dict[c] if c in correct_dict else c for c in cudas]
    return ','.join(cudas)


def main():
    args, task_args, model_args, trainer_args = parse_cmd_args(task_name='Task')
    args.cuda = correct_cuda_no(args.cuda)
    task_args.cuda = correct_cuda_no(task_args.cuda)
    task = Task(**vars(task_args), model_args=vars(model_args), trainer_args=vars(trainer_args))
    task.run()
    return


if __name__ == '__main__':
    main()
