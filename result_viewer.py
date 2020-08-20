#!/usr/bin/env python3

from argparse import ArgumentParser
from viewer.viewer import Viewer
import pandas as pd
import sys
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("csv_file", action="store", help="Path to the CSV file with results")

    options = parser.parse_args()
    data = pd.read_csv(options.csv_file, index_col=0)
    viewer = Viewer(sys.argv, os.path.abspath(options.csv_file).replace(".csv", ""), data)
    viewer.run()

