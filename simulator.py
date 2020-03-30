import argparse
import json


class GraphManager():
    def __init__(self):
        pass


class Simulator():
    def __init__(self, config_filename):
        self.parseConfig(config_filename)


    def parseConfig(self, config_filename):
        with open(config_filename) as f:
            self.parameters = json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", 
        help="The configuration file to use for the simulation.",
        required=True)

    args = parser.parse_args()

    sim = Simulator(args.config)