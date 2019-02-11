from multiprocessing import Process
import argparse
import functools

from vz.asr.base.encoders import ProbCodec

from data_utils.data import DataGenerator
from utils.utility import add_arguments
from util import infer_batch


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('num_samples', int, 10, "# of samples to infer.")
add_arg('trainer_count', int, 8, "# of Trainers (CPUs or GPUs).")
add_arg('num_conv_layers', int, 2, "# of convolution layers.")
add_arg('num_rnn_layers', int, 3, "# of recurrent layers.")
add_arg('rnn_layer_size', int, 2048, "# of recurrent cells per layer.")
add_arg('use_gru', bool, False, "Use GRUs instead of simple RNNs.")
add_arg('use_gpu', bool, True, "Use GPU or not.")
add_arg('share_rnn_weights', bool, True, "Share input-hidden weights across "
                                         "bi-directional RNNs. Not for GRU.")
add_arg('infer_manifest', str,
        'data/librispeech/manifest.dev-clean',
        "Filepath of manifest to infer.")
add_arg('mean_std_path', str,
        'data/librispeech/mean_std.npz',
        "Filepath of normalizer's mean & std.")
add_arg('vocab_path', str,
        'data/librispeech/vocab.txt',
        "Filepath of vocabulary.")
add_arg('model_path', str,
        './checkpoints/libri/params.latest.tar.gz',
        "If None, the training starts from scratch, "
        "otherwise, it resumes from the pre-trained model.")
add_arg('specgram_type', str,
        'linear',
        "Audio feature type. Options: linear, mfcc.",
        choices=['linear', 'mfcc'])
# yapf: disable
args = parser.parse_args()


def print_func(continent='Asia'):
    print('The name of continent is : ', continent)

if __name__ == "__main__":
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
    cols = ['filepath', 'orig_script', 'infer_script', 'encode']

    for batch_id, data_batch in enumerate(batch_reader()):
        infer_args = (data_batch, data_generator, codec, batch_id, args)
        proc = Process(target=infer_batch, args=infer_args)
        proc.start()
        proc.join()

