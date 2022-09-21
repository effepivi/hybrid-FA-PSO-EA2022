ALGORITHM=CCPSO

for i in {1..10}
do
	echo $ALGORITHM-$i

	echo "#!/bin/bash"                                                     > submit-$ALGORITHM-$i.sh
	echo "#"                                                               >> submit-$ALGORITHM-$i.sh
	echo "#"                                                               >> submit-$ALGORITHM-$i.sh
	echo "#SBATCH --job-name=$ALGORITHM-$i     # Job name"                 >> submit-$ALGORITHM-$i.sh
	echo "#SBATCH --nodes=1                    # Use one node"             >> submit-$ALGORITHM-$i.sh
	echo "#SBATCH --ntasks-per-node=1          # Number of tasks per node" >> submit-$ALGORITHM-$i.sh
	echo "#SBATCH --cpus-per-task=1            # Number of cores per task" >> submit-$ALGORITHM-$i.sh
	echo "#SBATCH --mem=600mb                  # Total memory limit"       >> submit-$ALGORITHM-$i.sh
	echo "#SBATCH --time=72:00:00              # Time limit hrs:min:sec"   >> submit-$ALGORITHM-$i.sh
	echo "#SBATCH --account=scw1626"                                       >> submit-$ALGORITHM-$i.sh
	echo "#"                                                               >> submit-$ALGORITHM-$i.sh
	echo "#"                                                               >> submit-$ALGORITHM-$i.sh
        echo "source ./env.sh"                                                 >> submit-$ALGORITHM-$i.sh
	echo "./run-$ALGORITHM-$i.sh"                                          >> submit-$ALGORITHM-$i.sh

	sbatch submit-$ALGORITHM-$i.sh
done
