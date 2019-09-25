#!/bin/bash

# Usage: provide the experiment IDs as arguments with optional glomming.
# Example:
#   $ ./run.sh lrt-020-00[7-9] lrt-020-01* lrt-020-02[0-3]
# This will run experiments lrt-020-007 through lrt-020-023.

exps=$(python3 -c "from lr.experiments import configs; print(' '.join(sorted(configs.keys())))")

echo "List of experiments this script will run:"
for arg; do
    for exp in $exps; do
        if [[ $exp == $arg ]]; then
            echo $exp
        fi
    done
done

echo ""
echo "********************************************************************************"
echo ""

for arg; do
    for exp in $exps; do
        if [[ $exp == $arg ]]; then
            echo "Running: $exp"
            python lr/main.py --name=$exp --pdb=0
        fi
    done
done


