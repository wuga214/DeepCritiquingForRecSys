from utils.io import load_yaml, get_file_names, write_file

import argparse
import os


def main(args):
    bash_path = load_yaml('config/global.yml', key='path')['bashes']

    config_path = 'config/{}'.format(args.dataset_name)
    yaml_files = get_file_names(config_path, extension='.yml')
    project_path = "~/DeepCritiquingForRecSys"

    pattern = "#!/usr/bin/env bash\n" \
              "source {0}\n" \
              "cd {1}\n" \
              "python tune_parameters.py --data_dir {2} --save_path {3}/{4}.csv --parameters config/{3}/{4}.yml\n"

    for setting in yaml_files:
        name, extension = os.path.splitext(setting)
        content = pattern.format(args.virtualenv_path, project_path, args.data_dir, args.dataset_name, name)
        write_file(bash_path+args.dataset_name, args.dataset_name+'-'+name+'.sh', content, exe=True)

    bash_files = sorted(get_file_names(bash_path+args.dataset_name, extension='.sh'))

    commands = []
    command_pattern = 'sbatch --nodes=1 --time={0}:00:00 --mem={1} --cpus=4 '
    if args.gpu:
        command_pattern = command_pattern + '--gres=gpu:1 '

    command_pattern = command_pattern + '{2}'

    for bash in bash_files:
        commands.append(command_pattern.format(args.max_time, args.memory, bash))
    content = "\n".join(commands)
    write_file(bash_path + args.dataset_name, 'run_' + args.dataset_name + '.sh', content)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Create Tuning Bash")

    parser.add_argument('--dataset_name', dest='dataset_name', default="CDsVinyl")
    parser.add_argument('--virtualenv_path', dest='virtualenv_path', default='~/ENV/bin/activate')
    parser.add_argument('--data_dir', dest='data_dir', default="data/CDsVinyl/")
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--max_time', dest='max_time', default='72')
    parser.add_argument('--memory', dest='memory', default='32G')

    args = parser.parse_args()

    main(args)
