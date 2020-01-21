# prepare folder
if [ ! -e data/custom ]; then
    mkdir data/custom
fi


head -n 64  /nfs/alldata/Airtel/Manifest/pipeline/inference_outputs/manifest.airtel  > data/custom/manifest.tiny

# build vocabulary
python tools/build_vocab.py \
--count_threshold=0 \
--vocab_path='data/custom/vocab.txt' \
--manifest_paths='data/custom/manifest.tiny'

if [ $? -ne 0 ]; then
    echo "Build vocabulary failed. Terminated."
    exit 1
fi


# compute mean and stddev for normalizer
python tools/compute_mean_std.py \
--manifest_path='data/custom/manifest.tiny' \
--num_samples=64 \
--specgram_type='linear' \
--output_path='data/custom/mean_std.npz'

if [ $? -ne 0 ]; then
    echo "Compute mean and stddev failed. Terminated."
    exit 1
fi


echo "Custom Data preparation done."
exit 0
