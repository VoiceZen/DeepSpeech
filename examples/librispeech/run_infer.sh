#! /usr/bin/env bash

cd ../.. > /dev/null

# download language model
# cd models/lm > /dev/null
# sh download_lm_en.sh
if [ $? -ne 0 ]; then
    exit 1
fi
# cd - > /dev/null


# infer
CUDA_VISIBLE_DEVICES=0,1 \
python -u greed_infer.py \
--num_samples=32 \
--trainer_count=2 \
--num_conv_layers=2 \
--num_rnn_layers=3 \
--rnn_layer_size=1024 \
--use_gru=True \
--use_gpu=False \
--share_rnn_weights=True \
--infer_manifest='/tmp/manifest.test' \
--mean_std_path='/asr/models/trained/mean_std.npz' \
--vocab_path='/asr/models/trained/vocab.txt' \
--model_path='/asr/models/trained/params.tar.gz' \
--decoding_method='ctc_greedy' \
--error_rate_type='wer'

if [ $? -ne 0 ]; then
    echo "Failed in inference!"
    exit 1
fi


exit 0