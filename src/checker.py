#!/usr/bin/env python3

import sys
import os
import re

# Usage: ./checker.py -naive/-pisinger/-openmp


if len(sys.argv) != 4:
    print("Usage: ./checker.py scene threads -pisinger/-openmp")
    exit(-1)

scene = sys.argv[1] # 50k_1k_10k
threads = sys.argv[2]
version = sys.argv[3]

valid = True
valid = valid and (version == '-pisinger' or version == '-openmp')

if not valid:
    print("Usage: ./checker.py threads scene -pisinger/-openmp")
    exit(-1)


prog = ""
if version=='-pisinger':
    prog = "pisinger_test"
elif version=='-openmp':
    prog = "parallel_cpu_test"

# set num threads
os.environ['OMP_NUM_THREADS'] = threads


def compare(actual, ref):
    actual = open(actual).readlines()
    ref = open(ref).readlines()
    assert actual[0]==ref[0], 'ERROR -- incorrect subset sum result'


os.system('mkdir -p logs')
os.system('rm -rf logs/*')
output_file = f"logs/{scene}_{prog}_{threads}.log"
cmd = f'bin/{prog} < inputs/{scene} > {output_file}'
ret = os.system(cmd)
assert ret == 0, f'ERROR -- {prog} exited with errors'


# run naive
naive_file = f"logs/{scene}_naive_test.log"
cmd = f'bin/naive_test < inputs/{scene} > {naive_file}'
ret = os.system(cmd)
assert ret == 0, f'ERROR -- naive_test exited with errors'

compare(output_file, naive_file)
t = float(open(output_file).readlines()[1])
naive_t = float(open(naive_file).readlines()[1])
print(f'total simulation time: {t:.6f}s')
print(f'speedup over naive: {naive_t/t:.6f}')