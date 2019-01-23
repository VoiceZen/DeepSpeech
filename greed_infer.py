"""Inferer for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import paddle.v2 as paddle
from data_utils.data import DataGenerator
from model_utils.model import DeepSpeech2Model
from utils.utility import add_arguments, print_arguments
from util import infer_batch
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

    ds2_model = DeepSpeech2Model(
        vocab_size=data_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_layer_size=args.rnn_layer_size,
        use_gru=args.use_gru,
        pretrained_model_path=args.model_path,
        share_rnn_weights=args.share_rnn_weights)

    vocab_list = [chars.encode("utf-8") for chars in data_generator.vocab_list]
    codec = ProbCodec(vocab_list, True)
    iterations_for_write = 2  # 200
    res = []
    cols = ['filepath', 'orig_script', 'infer_script', 'encode']
    for batch_id, data_batch in enumerate(batch_reader()):
        print("batch id is {}".format(batch_id))
        print((batch_id + 1) % iterations_for_write)
        if (batch_id + 1) % iterations_for_write == 0:
            # write to csv
            # this is just to ensure oom
            df = pd.DataFrame(res, columns=cols)
            tmp = int((batch_id + 1) / iterations_for_write)
            df.to_csv(
                'batch-infer-{}.csv'.format(tmp),
                index=False,
                header=False
            )
            res = []

        infer_out = infer_batch(
            ds2_model, data_batch, data_generator, codec, args)
        res.extend(infer_out)

    df = pd.DataFrame(res, columns=cols)
    df.to_csv(
        'batch-infer-last.csv'.format(batch_id),
        index=False,
        header=False
    )
    import pdb; pdb.set_trace()  # breakpoint 57d9f2d3 //



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
