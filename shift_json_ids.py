import json
import argparse


def shift_ids(input_file, output_file):
    with open(input_file) as json_file:
        predictions_array = json.load(json_file)

    for pred in predictions_array:
        pred['category_id'] += 1

    with open(output_file, 'w') as json_file:
        json.dump(predictions_array, json_file, indent=4, sort_keys=False)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='A script to shift class_ids for CVAT annotations')
	parser.add_argument("input_path")
	parser.add_argument("output_path")
	args = parser.parse_args()
	input_file = args.input_path
	output_file = args.output_path
	shift_ids(input_file, output_file)
