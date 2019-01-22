"""Inferer for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import paddle.v2 as paddle
from data_utils.data import DataGenerator
from model_utils.model import DeepSpeech2Model
from utils.error_rate import wer, cer
from utils.utility import add_arguments, print_arguments
import pandas as pd

from vz.asr.base.encoders import ProbCodec

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('num_samples',      int,    10,     "# of samples to infer.")
add_arg('trainer_count',    int,    8,      "# of Trainers (CPUs or GPUs).")
add_arg('beam_size',        int,    500,    "Beam search width.")
add_arg('num_proc_bsearch', int,    8,      "# of CPUs for beam search.")
add_arg('num_conv_layers',  int,    2,      "# of convolution layers.")
add_arg('num_rnn_layers',   int,    3,      "# of recurrent layers.")
add_arg('rnn_layer_size',   int,    2048,   "# of recurrent cells per layer.")
add_arg('use_gru',          bool,   False,  "Use GRUs instead of simple RNNs.")
add_arg('use_gpu',          bool,   True,   "Use GPU or not.")
add_arg('share_rnn_weights',bool,   True,   "Share input-hidden weights across "
                                            "bi-directional RNNs. Not for GRU.")
add_arg('infer_manifest',   str,
        'data/librispeech/manifest.dev-clean',
        "Filepath of manifest to infer.")
add_arg('mean_std_path',    str,
        'data/librispeech/mean_std.npz',
        "Filepath of normalizer's mean & std.")
add_arg('vocab_path',       str,
        'data/librispeech/vocab.txt',
        "Filepath of vocabulary.")
add_arg('model_path',       str,
        './checkpoints/libri/params.latest.tar.gz',
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

    vocab_list = [chars.encode("utf-8") for chars in data_generator.vocab_list]

    codec = ProbCodec(vocab_list, True)
    infer_data = batch_reader().next()

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

    ds2_model.logger.info("start inference ...")

    probs_split = ds2_model.infer_batch_probs(
        infer_data=infer_data,
        feeding_dict=data_generator.feeding)

    result_transcripts = ds2_model.decode_batch_greedy(
        probs_split=probs_split,
        vocab_list=vocab_list)

    result_encodings = ds2_model.get_encoded_strings(
        probs_split=probs_split,
        codec=codec
    )

    # error_rate_func = cer if args.error_rate_type == 'cer' else wer
    # target_transcripts = [data[1] for data in infer_data]

    cols = ['filepath', 'orig_script', 'infer_script', 'encode']
    res = []
    for target, result, encode in zip(
        infer_data,
        result_transcripts,
        result_encodings
    ):
        res.append([
            target[3],
            target[1],
            result,
            encode
        ])

    df = pd.DataFrame(res, columns=cols)
    df.to_csv('infer.csv', index=False)
    ds2_model.logger.info("finish inference")


def main():
    print_arguments(args)
    if args.use_gpu:
        paddle.init(
            use_gpu=True,
            rnn_use_batch=True,
            trainer_count=args.trainer_count)
    else:
        paddle.init(use_gpu=False, trainer_count=1)
    infer()

if __name__ == '__main__':
    main()
