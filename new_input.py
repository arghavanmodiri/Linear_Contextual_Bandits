from new_main import main_algo
import numpy as np
import argparse
import sys
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('input_file', metavar='input_file', type=str, nargs=1,
                        help='Name of the json config file')
    args = parser.parse_args()

    if (len(args.input_file) != 1) or (not args.input_file[0].endswith(".json")):
        print( "Error: Function should have only one input, name of the JSON config file." )
        sys.exit(1)

    input_data = args.input_file[0]
    input_data = json.load(open(input_data))
    main_algo(input_data)


