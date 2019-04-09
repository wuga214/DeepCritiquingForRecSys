#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/InterpretableAutoRec
python progress_analysis.py --explanation -n explanation_convergence_analysis2.csv -e 1000
