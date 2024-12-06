#!/bin/sh
cd $HOME/lab/forked_repo/DyGLib/scripts

for K in 2
do
    for N in 2 4 8 16 32 64 128 256 512
    do
        for SCRIPT in "_dygformer" "_tgat" "_tgn"
        do
            DATASET="($K, $N)"
            
            MEM=$(echo "4 * $K * $N / 16" | bc)
            MEM=$(printf "%.0f" "$MEM")
            if [ $(echo "$MEM < 4" | bc) -eq 1 ]; then
                MEM=4
            fi

            sbatch \
                --output="logs/($K, $N)/$SCRIPT/%j-o.out" \
                --error="logs/($K, $N)/$SCRIPT/%j-e.out" \
                --mem=${MEM}gb \
                --gres=gpu:48gb:1 \
                $SCRIPT.sh "$DATASET"
        done
    done
done
