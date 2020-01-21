
CUDA_VISIBLE_DEVICES=0,1 \
python -u infer.py \
--num_samples=10 \
--beam_size=500 \
--num_proc_bsearch=8 \
--num_conv_layers=2 \
--num_rnn_layers=3 \
--rnn_layer_size=2048 \
--alpha=2.5 \
--beta=0.3 \
--cutoff_prob=1.0 \
--cutoff_top_n=40 \
--use_gru=False \
--use_gpu=True \
--share_rnn_weights=True \
--infer_manifest='/home/vz/Users/ak47/projects/DeepSpeech/sample.csv' \
--mean_std_path='data/custom/mean_std.npz' \
--vocab_path='data/custom/vocab.txt' \
--model_path='/nfs/alldata/zeus/pretrained_ds2/libris' \
--lang_model_path='/nfs/alldata/zeus/pretrained_ds2/common_crawl_00.prune01111.trie.klm' \
--decoding_method='ctc_beam_search' \
--error_rate_type='wer' \
--specgram_type='linear'


if [ $? -ne 0 ]; then
    echo "Failed in inference!"
    exit 1
fi


exit 0

