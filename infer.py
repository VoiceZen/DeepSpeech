"""Inferer for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import json
import codecs
import functools
import pickle
import pandas as pd
import paddle.v2 as paddle
from data_utils.data import DataGenerator
from model_utils.model import DeepSpeech2Model
from utils.error_rate import wer, cer
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('num_samples',      int,    10,     "# of samples to infer.")
add_arg('trainer_count',    int,    16,      "# of Trainers (CPUs or GPUs).")
add_arg('beam_size',        int,    500,    "Beam search width.")
add_arg('num_proc_bsearch', int,    8,      "# of CPUs for beam search.")
add_arg('num_conv_layers',  int,    2,      "# of convolution layers.")
add_arg('num_rnn_layers',   int,    3,      "# of recurrent layers.")
add_arg('rnn_layer_size',   int,    2048,   "# of recurrent cells per layer.")
add_arg('alpha',            float,  2.5,    "Coef of LM for beam search.")
add_arg('beta',             float,  0.3,    "Coef of WC for beam search.")
add_arg('cutoff_prob',      float,  1.0,    "Cutoff probability for pruning.")
add_arg('cutoff_top_n',     int,    40,     "Cutoff number for pruning.")
add_arg('use_gru',          bool,   False,  "Use GRUs instead of simple RNNs.")
add_arg('use_gpu',          bool,   False,   "Use GPU or not.")
add_arg('share_rnn_weights',bool,   True,   "Share input-hidden weights across "
                                            "bi-directional RNNs. Not for GRU.")
add_arg('infer_manifest',   str,
        'data/tiny/manifest.dev-clean',
        "Filepath of manifest to infer.")
add_arg('mean_std_path',    str,
        'data/tiny/mean_std.npz',
        "Filepath of normalizer's mean & std.")
add_arg('vocab_path',       str,
        'data/tiny/vocab.txt',
        "Filepath of vocabulary.")
add_arg('lang_model_path',  str,
        '/newmodel/common_crawl_00.prune01111.trie.klm',
        "Filepath for language model.")
add_arg('model_path',       str,
        '/home/ec2-user/priyanka/DeepSpeech/params.pass-9.tar.gz',
        "If None, the training starts from scratch, "
        "otherwise, it resumes from the pre-trained model.")
add_arg('decoding_method',  str,
        'ctc_beam_search',
        "Decoding method. Options: ctc_beam_search, ctc_greedy",
        choices = ['ctc_beam_search', 'ctc_greedy'])
add_arg('error_rate_type',  str,
        'wer',
        "Error rate type for evaluation.",
        choices=['wer', 'cer'])
add_arg('specgram_type',    str,
        'linear',
        "Audio feature type. Options: linear, mfcc.",
        choices=['linear', 'mfcc'])
add_arg('logits_file',    str,
        'linear',
        "Pickle file to  save the logits.",
        )
add_arg('infer_output_file', str,
        'output.csv',
        "path of output infer file")
# yapf: disable
args = parser.parse_args()


def infer():
    """Inference for DeepSpeech2."""
    data_generator = DataGenerator(
        vocab_filepath=args.vocab_path,
        mean_std_filepath=args.mean_std_path,
        augmentation_config='{}',
        specgram_type=args.specgram_type,
        num_threads=1,
        keep_transcription_text=True)
    batch_reader = data_generator.batch_reader_creator(
        manifest_path=args.infer_manifest,
        batch_size=args.num_samples,
        min_batch_size=1,
        sortagrad=False,
        shuffle_method=None)
    # infer_data = batch_reader().next()

    ds2_model = DeepSpeech2Model(
        vocab_size=data_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_layer_size=args.rnn_layer_size,
        use_gru=args.use_gru,
        pretrained_model_path=args.model_path,
        share_rnn_weights=args.share_rnn_weights)

    # decoders only accept string encoded in utf-8
    vocab_list = [chars.encode("utf-8") for chars in data_generator.vocab_list]
    manifest = args.infer_manifest
    df = pd.read_csv(manifest)
    total_batches = int(df.shape[0]/args.num_samples)
    manifest = []
    for json_line in codecs.open(args.infer_manifest, 'r', 'utf-8'):
        try:
            json_data = json.loads(json_line)
        except Exception as e:
            raise IOError("Error reading manifest: %s" % str(e))
    
        manifest.append(json_data)
    final_man=pd.DataFrame(manifest)
    dump_out = {}
    dump_results = {}
    res=[]
    ds2_model.logger.info("start inference ...")

    for index, infer_data in enumerate(batch_reader()):
        print("----------------------")
        if index % 5 == 0:
            ds2_model.logger.info("Completed {}/{}".format(index, total_batches))

        probs_split = ds2_model.infer_batch_probs(infer_data=infer_data,
            feeding_dict=data_generator.feeding)

        for index, input_data in enumerate(infer_data):
            _, audio_path, _ ,_= input_data
            dump_results[audio_path] = probs_split[index]
        result_transcripts= ds2_model.decode_batch_beam_search(
            probs_split=probs_split,
            beam_alpha=args.alpha,
            beam_beta=args.beta,
            beam_size=args.beam_size,
            cutoff_prob=args.cutoff_prob,
            cutoff_top_n=args.cutoff_top_n,
            vocab_list=vocab_list,
            num_processes=args.num_proc_bsearch)
        for j in result_transcripts:
            res.append(j)
        print(res)
    infer_preds = np.empty(shape=(df.shape[0], 2), dtype=object)
    infer_out={}
    for idx, line in enumerate(final_man['audio_filepath']):
        filename=line
        text = res[idx]#[v for v in zip(*res[idx])]
        #infer_preds[idx, 0] = filename
        #infer_preds[idx, 1] = text
        infer_out[filename]=text
    #np.savetxt(args.infer_output_file, infer_preds, fmt='%s', delimiter=',',header='wav_filename,lm')
    print(infer_out)

    dump_out["logits"] = dump_results
    dump_out["vocab"] = {
        index: char
        for index, char in enumerate(vocab_list)
    }
    # error_rate_func = cer if args.error_rate_type == 'cer' else wer
    # target_transcripts = [data[1] for data in infer_data]
    # for target, result in zip(target_transcripts, result_transcripts):
    #     print("\nTarget Transcription: %s\nOutput Transcription: %s" %
    #           (target, result))
    #     print("Current error rate [%s] = %f" %
    #           (args.error_rate_type, error_rate_func(target, result)))
    final_output_file=pd.DataFrame()
    final_output_file['audio_filepath']=list(infer_out.keys())
    final_output_file['predicted_text']=list(infer_out.values())
    final_output_file.to_csv(args.infer_output_file,index=False)
    with open(args.logits_file, 'wb') as f:
        pickle.dump(dump_out, f, protocol=pickle.HIGHEST_PROTOCOL)
    ds2_model.logger.info("finish inference")

def main():
    print_arguments(args)
    paddle.init(use_gpu=args.use_gpu,
                rnn_use_batch=True,
                trainer_count=args.trainer_count)
    infer()


if __name__ == '__main__':
    main()
