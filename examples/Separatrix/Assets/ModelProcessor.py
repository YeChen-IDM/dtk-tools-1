import os
import math
import random
import json


dir_path = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(dir_path, "..", "output")

def displayArguments(arguments):
    print('Input config (JSON):       %s' % arguments.config)


def test_func(x, y):
    return random.random() < .25 * (math.tanh(10 * x - 5) + 1) * (math.tanh(3 * y - 0.9) + 1)


def main():
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    config = json.load(open(os.path.join(dir_path, "..", "config.json"), "r"))
    # print(config)

    result = {'Point_X': config['Point_X'], 'Point_Y': config['Point_Y'],
              '__sample_index__': config['__sample_index__'],
              'Result': 1 if test_func(config['Point_X'], config['Point_Y']) else 0}

    with open(os.path.join(output_dir, "result.json"), "w") as fp:
        json.dump(result, fp)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', help="Config (JSON))", required=True)
    # args = parser.parse_args()
    # displayArguments(args)
    #
    # main(args.config)

    main()
