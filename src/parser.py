
from src.config import Config
import argparse
import sys
import os

class Parser:

    def __init__(self):
        # parse arguments
        self.parser = argparse.ArgumentParser(description='script.')

    def update_config(self, config):
        # import default configuration
        
        # Add arguments
        attributes_list = list(vars(config).keys())
        for at in attributes_list:
            self.parser.add_argument(f'--{at}', type=float, help=at,default=None )

        # Parse the command-line arguments
        args = self.parser.parse_args()

        for at in attributes_list:
            new_value=getattr(args, at)
            if new_value is not None:
                setattr(config, at, new_value)
            config.derived_variables()
            
        print('******************')
        print('Config:')
        new_attributes_list = list(vars(config).keys())
        for at in new_attributes_list:
            print(f'{at}: {getattr(config, at)}')
        print('******************')

        return config