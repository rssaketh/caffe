#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=5000
#SBATCH --gres=gpu:1
#SBATCH -t 00:20:00
#SBATCH -p gpu


set -e
source load_caffe_dependencies.sh
./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt $@
