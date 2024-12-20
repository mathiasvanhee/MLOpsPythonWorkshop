import argparse
from pathlib import Path
from test import execute_model_and_generate_integration_test_data

import logging

parser = argparse.ArgumentParser("test")
parser.add_argument("--model_input", type=str)
parser.add_argument("--images_input", type=str)
parser.add_argument("--model_output", type=str)

# Get arguments from parser
args = parser.parse_args()
model_input = args.model_input
images_input = args.images_input
model_output = args.model_output

statistics = execute_model_and_generate_integration_test_data(logging,
                                                              Path(model_input),
                                                              Path(images_input),
                                                              Path(model_output))
