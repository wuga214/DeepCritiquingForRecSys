#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/InterpretableAutoRec
python reproduce_critiquing.py -d data/beer/ -p beer -s beer_Critiquing5.csv -f beer_Falling_Rank5
