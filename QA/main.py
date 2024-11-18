import time
import argparse
from utils.arguments import ArgParser
from utils.utils import check_path, set_random_seed

def main(args: argparse.Namespace) -> None:
    if args.seed is not None:
        set_random_seed(args.seed)

    start_time = time.time()

    for path in []:
        check_path(path)

    if args.job is None:
        raise ValueError('Please specify the job to do.')
    else:
        if args.task == 'question_answering':
            if args.job == 'preprocessing':
                from task.question_answering.preprocessing import preprocessing as job
            elif args.job in ['training', 'resume_training']:
                from task.question_answering.train import training as job
            elif args.job == 'testing':
                from task.question_answering.test import testing as job
            else:
                raise ValueError(f'Invalid job: {args.job}')
        else:
            raise ValueError(f'Invalid task: {args.task}')

    job(args)

    elapsed_time = time.time() - start_time
    print(f'Completed {args.job}; Time elapsed: {elapsed_time / 60:.2f} minutes')

if __name__ == '__main__':
    parser = ArgParser()
    args = parser.get_args()

    main(args)