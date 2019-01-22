#! /usr/bin/env bash

cd ../.. > /dev/null

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
--infer_manifest='data/librispeech/manifest.test-clean' \
--mean_std_path='data/librispeech/mean_std.npz' \
--vocab_path='data/librispeech/vocab.txt' \
--model_path='checkpoints/libri/step_final' \
--lang_model_path='models/lm/common_crawl_00.prune01111.trie.klm' \
--decoding_method='ctc_beam_search' \
--error_rate_type='wer' \
--specgram_type='linear'

if [ $? -ne 0 ]; then
    echo "Failed in inference!"
    exit 1
fi


exit 0