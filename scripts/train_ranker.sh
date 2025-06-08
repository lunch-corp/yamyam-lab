for type in popularity random
do
    for n in 10 30 50 100
    do
        python src/train_ranker.py \
            models/ranker=lightgbm \
            data.num_neg_samples=$n \
            data.sampling_type=$type
    done
done
