from simba.config import Config
import argparse
import sys
import os


class Parser:

    def __init__(self):
        # parse arguments
        self.parser = argparse.ArgumentParser(description="script.")

    def update_config(self, config):
        # import default configuration

        # Add arguments
        attributes_list = list(vars(config).keys())

        for at in attributes_list:
            is_integer_attribute = (
                (at == "BATCH_SIZE")
                or (at == "N_LAYERS")
                or (at == "D_MODEL")
                or (at == "epochs")
                or (at == "PREPROCESSING_NUM_NODES")
                or (at == "PREPROCESSING_CURRENT_NODE")
                or (at == "PREPROCESSING_NUM_WORKERS")
                or (at == "TRAINING_NUM_WORKERS")
            )

            is_string_attribute = at in [
                "ACCELERATOR",
                "extra_info",
                "dataset_path",
                "PREPROCESSING_DIR",
                "BEST_MODEL_NAME",
                "PRETRAINED_MODEL_NAME",
                "CHECKPOINT_DIR",
                "PREPROCESSING_DIR_TRAIN",
                "PREPROCESSING_DIR_TEST",
                "pretrained_path",
            ]
            if is_integer_attribute:
                self.parser.add_argument(f"--{at}", type=int, help=at, default=None)
            elif is_string_attribute:
                self.parser.add_argument(f"--{at}", type=str, help=at, default=None)
            else:
                self.parser.add_argument(f"--{at}", type=float, help=at, default=None)

        # Parse the command-line arguments
        args = self.parser.parse_args()

        for at in attributes_list:
            new_value = getattr(args, at)
            if new_value is not None:
                setattr(config, at, new_value)
            config.derived_variables()

        print("******************")
        print("Config:")
        new_attributes_list = list(vars(config).keys())
        for at in new_attributes_list:
            print(f"{at}: {getattr(config, at)}")
        print("******************")

        return config
