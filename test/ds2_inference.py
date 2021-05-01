"""Inferer for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('..')
sys.path.append('/home/hitler/users/interns/kuldip/DeepSpeech/')
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
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


asr_config = json.load(open('../service/ds2_base_config.json'))
args = dotdict(asr_config['config'])

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
 
    df = pd.read_csv(args.infer_manifest)
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

    infer_preds = np.empty(shape=(df.shape[0], 2), dtype=object)
    infer_out={}
    for idx, line in enumerate(final_man['audio_filepath']):
        filename=line
        text = res[idx]
        infer_out[filename]=text
    
    dump_out["logits"] = dump_results
    dump_out["vocab"] = {
        index: char
        for index, char in enumerate(vocab_list)
    }
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
