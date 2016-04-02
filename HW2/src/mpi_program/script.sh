#!/bin/bash
#SBATCH -J Dist_Sort      # Job Name
#SBATCH -o Output.o%j    # Output and error file name (%j expands to jobID)
#SBATCH -n 32           # Total number of  tasks requested
#SBATCH -p development  # Queue (partition) name -- normal, development, etc.
#SBATCH -t 0:01:00     # Run time (hh:mm:ss)
#SBATCH --mail-user=aadarshkenia@gmail.com

ibrun dist_sort 64 > log


