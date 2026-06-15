#!/bin/bash
#SBATCH --account=SC-001095
#SBATCH --job-name=pramsbbgo_run8b4_mainwrapper
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5-00:00:00
#SBATCH --mem-per-cpu=12000

module load python
PORT=4020
restart=False

for iteration in {0..20}; do
    # Outer iteration
    if [ "$restart" != "True" ]; then
        echo "Submitting master instance for outer iteration $iteration at $(date)"
        
        rm -rf master
        rm -rf template_outer.tar.gz

        cp -r template_outer/ master/
        tar -czf template_outer.tar.gz template_outer/

        MASTER_OUTER_JOB_ID=$(sbatch --parsable outer_raisemaster.sbatch)
        echo "$(date '+%Y-%m-%d %H:%M:%S') Master job submitted with ID: $MASTER_OUTER_JOB_ID"

        # Wait for the master job to start and get its assigned node
        echo "$(date '+%Y-%m-%d %H:%M:%S') Waiting for master job to start..."
        MASTER_NODE=""
        while [ -z "$MASTER_NODE" ]; do
            sleep 5
            # Get the node name where the master job is running
            MASTER_NODE=$(squeue -j $MASTER_OUTER_JOB_ID -h -o "%N" 2>/dev/null | grep -v "^$")
            
            # Check if job failed to start
            if ! squeue -j $MASTER_OUTER_JOB_ID &>/dev/null && [ -z "$MASTER_NODE" ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') ERROR: Master job failed to start"
                exit 1
            fi
        done

        echo "$(date '+%Y-%m-%d %H:%M:%S') Master outer is running on node: $MASTER_NODE"
        # Update outer_runworker with the actual master node name
        sed -i -E "s|/H [^:]*:4[0-9]{3}|/H ${MASTER_NODE}:${PORT}|g" outer_runworker
        
        # Wait a bit more for master to be fully ready
        echo "$(date '+%Y-%m-%d %H:%M:%S') Waiting for outer master to be ready..."
        sleep 10
        
        # Check if master is still running
        if ! squeue -j $MASTER_OUTER_JOB_ID &>/dev/null; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') ERROR: Master job is no longer in queue"
            continue
        fi

        # if [ $iteration -eq 0 ]; then
        #     sed -i "s/#SBATCH --array=0-24/#SBATCH --array=0-99/g" outer_iter.sbatch
        #     sed -i "s/#SBATCH --ntasks-per-node=4/#SBATCH --ntasks-per-node=5/g" outer_iter.sbatch
        # elif [ $iteration -eq 1 ]; then
        #     sed -i "s/#SBATCH --array=0-99/#SBATCH --array=0-24/g" outer_iter.sbatch
        #     sed -i "s/#SBATCH --ntasks-per-node=5/#SBATCH --ntasks-per-node=4/g" outer_iter.sbatch
        # fi
        
        echo "$(date '+%Y-%m-%d %H:%M:%S') Submitting outer iteration workers for iteration $iteration"
        OUTER_JOB_ID=$(sbatch --parsable outer_iter.sbatch)
        echo "Outer iteration $iteration submitted with job ID: $OUTER_JOB_ID"

        # Wait for outer iteration to complete
        echo "$(date '+%Y-%m-%d %H:%M:%S') Waiting for outer iteration to complete..."
        while squeue -j $OUTER_JOB_ID &>/dev/null; do
            sleep 60
        done

        echo "$(date '+%Y-%m-%d %H:%M:%S') Waiting for outer master job to complete..."
        while squeue -j $MASTER_OUTER_JOB_ID &>/dev/null; do
            sleep 60
        done

        echo "$(date '+%Y-%m-%d %H:%M:%S') Outer iteration completed"

        # Rename the "master" directory to "outer"
        if [ -d "master" ]; then
            mv master outer_$iteration
        fi
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') Skipping outer iteration setup for iteration $iteration (restart mode)"
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') Running interproc.py for outer iteration $iteration"
    python -c "from interproc import intermediate_processing_1; intermediate_processing_1()"

    # Inner iteration
    echo "$(date '+%Y-%m-%d %H:%M:%S') Submitting master instance for inner iteration $((iteration+1)) at $(date)"

    rm -rf master
    rm -rf template_inner.tar.gz

    cp -r template_inner/ master/
    tar -czf template_inner.tar.gz template_inner/

    MASTER_INNER_JOB_ID=$(sbatch --parsable inner_raisemaster.sbatch)
    echo "$(date '+%Y-%m-%d %H:%M:%S') Master job submitted with ID: $MASTER_INNER_JOB_ID"

    # Wait for the master job to start and get its assigned node
    echo "$(date '+%Y-%m-%d %H:%M:%S') Waiting for inner master job to start..."
    MASTER_NODE=""
    while [ -z "$MASTER_NODE" ]; do
        sleep 5
        # Get the node name where the master job is running
        MASTER_NODE=$(squeue -j $MASTER_INNER_JOB_ID -h -o "%N" 2>/dev/null | grep -v "^$")
        
        # Check if job failed to start
        if ! squeue -j $MASTER_INNER_JOB_ID &>/dev/null && [ -z "$MASTER_NODE" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') ERROR: Master job failed to start"
            exit 1
        fi
    done
    echo "$(date '+%Y-%m-%d %H:%M:%S') Master inner is running on node: $MASTER_NODE"
    # Update inner_runworker with the actual master node name
    sed -i -E "s|/H [^:]*:4[0-9]{3}|/H ${MASTER_NODE}:${PORT}|g" inner_runworker

    echo "$(date '+%Y-%m-%d %H:%M:%S') Submitting inner iteration workers for iteration $((iteration+1))"
    INNER_JOB_ID=$(sbatch --parsable inner_iter.sbatch)
    echo "Inner iteration $((iteration+1)) submitted with job ID: $INNER_JOB_ID"

    # Wait for inner iteration to complete
    echo "$(date '+%Y-%m-%d %H:%M:%S') Waiting for inner iteration to complete..."
    while squeue -j $INNER_JOB_ID &>/dev/null; do
        sleep 60
    done

    echo "$(date '+%Y-%m-%d %H:%M:%S') Waiting for inner master job to complete..."
    while squeue -j $MASTER_INNER_JOB_ID &>/dev/null; do
        sleep 60
    done

    echo "$(date '+%Y-%m-%d %H:%M:%S') Inner iteration completed"
    # Rename the "master" directory to "inner"
    if [ -d "master" ]; then
        mv master inner_$((iteration+1))
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') Running interproc.py for inner iteration $((iteration+1))"
    python -c "from interproc import intermediate_processing_2; intermediate_processing_2()"

    if [ "$restart" = "True" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') Restart mode completed. Switching to normal mode for subsequent iterations."
        restart=False
    fi
done

echo "$(date '+%Y-%m-%d %H:%M:%S') All iterations completed at $(date)"