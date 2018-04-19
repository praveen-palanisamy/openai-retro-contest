#!/usr/bin/env python
"""
A script to take the sonic-train.csv and sonic-validation.csv files and generate
the sonic_config.json file with the train/val split used in experiments (For eg.: GA3C)
"""
import pandas as pd
import argparse
import json

args = argparse.ArgumentParser(description="Sonic Conf generator")
args.add_argument(
    '--game',
    type=str,
    default='SonicTheHedgehog-Genesis',
    metavar='G',
    help='Game to use. One among the retro Sonic game list. ALL will include all the 3 games' \
                                                        '. Default is SonicTheHedgehog-Genesis'
)
args = args.parse_args()

sonic_train = pd.read_csv("sonic-train.csv", header=0  )
sonic_val = pd.read_csv("sonic-validation.csv", header=0)

if args.game is not 'ALL':
    # input is not sanity checked
    sonic_train = sonic_train.where(sonic_train.game == args.game)
    sonic_val = sonic_val.where(sonic_val.game == args.game)

sonic_train_conf = sonic_train.apply(lambda row,: "{game:'" + row['game'] + "',level:'" +
                                              row['state'] + "'}", axis=1, ignore_failures=True)
sonic_val_conf = sonic_val.apply(lambda row,: "{game:'" + row['game'] + "',level:'" +
                                              row['state'] + "'}", axis=1, ignore_failures=True)

sonic_conf= {"Train": dict(sonic_train_conf), "Test": dict(sonic_val_conf) }
output_file = open("sonic_config.json", 'w')
json.dump(sonic_conf, fp=output_file, indent=4)


