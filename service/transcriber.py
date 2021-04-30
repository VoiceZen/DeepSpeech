import logging
import argparse
import functools
import os
import sys
import subprocess
import time
import paddle.v2 as paddle
from sys import platform
from multiprocessing import Process, Pipe
sys.path.append('..')
from data_utils.data import DataGenerator
from model_utils.model import DeepSpeech2Model
from data_utils.utility import read_manifest
from utils.utility import add_arguments, print_arguments
from asr_utils import ProbCodec, ScoringExperiment
from paddle.trainer.config_parser import logger as paddle_logger

class DeepSpeechTransriber(object):
    """Transcriber that integrates with DeepSpeech
    """
    # location to pick params, mean and vocab from
    root_fldr = '/asr/models/trained/'

    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('beam_size',        int,    500,    "Beam search width.")
    add_arg('num_proc_bsearch', int,    8,      "# of CPUs for beam search.")
    add_arg('num_conv_layers',  int,    2,      "# of convolution layers.")
    add_arg('num_rnn_layers',   int,    3,      "# of recurrent layers.")
    add_arg('rnn_layer_size',   int,    1024,   "# of recurrent cells per layer.")
    add_arg('alpha',            float,  1.4,   "Coef of LM for beam search.")
    add_arg('beta',             float,  0.35,   "Coef of WC for beam search.")
    add_arg('cutoff_prob',      float,  1.0,    "Cutoff probability for pruning.")
    add_arg('cutoff_top_n',     int,    40,     "Cutoff number for pruning.")
    add_arg('use_gru',          bool,   True,  "Use GRUs instead of simple RNNs.")
    add_arg('use_gpu',          bool,   False,   "Use GPU or not. Set to False for local use.")
    add_arg('trainer_count',    int,    16,   "Trainer count. Set to 1 for local use.")
    add_arg('share_rnn_weights',bool,   False,   "Share input-hidden weights across "
            "bi-directional RNNs. Not for GRU.")
    add_arg('mean_std_path',    str,
            root_fldr + 'mean_std.npz',
            "Filepath of normalizer's mean & std.")
    add_arg('vocab_path',       str,
            root_fldr + 'vocab.txt',
            "Filepath of vocabulary.")
    add_arg('model_path',       str,
            root_fldr + 'params.tar.gz',
            "If None, the training starts from scratch, "
            "otherwise, it resumes from the pre-trained model.")
    add_arg('lang_model_path',  str,
            '/asr/models/lm/art-trie.klm',
            "Filepath for language model.")
    add_arg('decoding_method',  str,
            'ctc_beam_search',
            "Decoding method. Options: ctc_beam_search, ctc_greedy",
            choices = ['ctc_beam_search', 'ctc_greedy'])
    add_arg('specgram_type',    str,
            'linear',
            "Audio feature type. Options: linear, mfcc.",
            choices=['linear', 'mfcc'])
    add_arg('audio_file', str, None, "Path to audio file")
    add_arg('log_level',  int, logging.INFO, "Logging level")
    add_arg('use_hindi',         bool,   False,  "Use hindi character translation of results .")
    # duplicated from runner, as we may run this separately which causes reinterpretation of args
    add_arg('test_manifest', str,   "./manifest.csv",  "Location of the test manifest")
    add_arg('out_path', str,   "./values.csv",  "Location to save the output .")
    add_arg('override_config', str, "./vz/config/overrides.json",  "Overrides for this run.")
    add_arg('custom_boost', bool, True,  "boosts from ctc.")
    add_arg('export_confidence', bool, False,  "Exports char probabilities.")
    add_arg('error_rate_type',  str,
        'wer',
        "Error rate type for evaluation.",
        choices=['wer', 'cer'])
    args = parser.parse_args()

    def __init__(self, max_retries = 5):
        self.logger = logging.getLogger(__name__)
        self.max_retries = max_retries
        self.worker = None
        self.master_conn = None
        self.logger.setLevel(DeepSpeechTransriber.args.log_level)

    def load_model(self):
        logger = logging.getLogger("DeepspeechTranscriber")
        paddle_logger.setLevel(logging.WARNING)
        args = DeepSpeechTransriber.args
        start_loading_model = time.time()
        paddle.init(
            use_gpu=args.use_gpu,
            rnn_use_batch=True,
            trainer_count=args.trainer_count)
        data_generator = DataGenerator(
            vocab_filepath=args.vocab_path,
            mean_std_filepath=args.mean_std_path,
            augmentation_config='{}',
            specgram_type=args.specgram_type,
            keep_transcription_text=False)
        ds2_model = DeepSpeech2Model(
            vocab_size=data_generator.vocab_size,
            num_conv_layers=args.num_conv_layers,
            num_rnn_layers=args.num_rnn_layers,
            rnn_layer_size=args.rnn_layer_size,
            use_gru=args.use_gru,
            pretrained_model_path=args.model_path,
            share_rnn_weights=args.share_rnn_weights)
        ds2_model.logger.setLevel(args.log_level)
        self.vocab_list = [chars.encode("utf-8") for chars in data_generator.vocab_list]
        self.ds2_model = ds2_model
        self.data_generator = data_generator 

        if not args.decoding_method == "ctc_greedy":
            # args.lang_model_path is treated as unicode, so convert to string else
            # swig will complain
            self.ds2_model.init_ext_scorer(
                args.alpha, args.beta, str(args.lang_model_path), self.vocab_list)
        end_loading_model = time.time()
        logger.info("Time to load model: {}".format(end_loading_model - start_loading_model))
        
    
    def set_args(self, args):
        DeepSpeechTransriber.args = args

    def shutdown(self):
        if self.master_conn:
            self.master_conn.send("exit")
            time.sleep(3)
            self.master_conn.close()

    def transcribe(self, file_paths):
        data_generator = self.data_generator
        ds2_model = self.ds2_model
        vocab_list = self.vocab_list
        exp = ScoringExperiment()
        codec = ProbCodec({itr: value for itr, value in enumerate(vocab_list)})
        logger = logging.getLogger("DeepspeechTranscriber")
        args = DeepSpeechTransriber.args
        logger.setLevel(args.log_level)
        st2 = time.time()
        manifest = [{
            "audio_filepath": file_path,
            "text": ""
        } for file_path in file_paths]

        st3 = time.time()
        reader, cleanup = data_generator._instance_reader_creator(manifest)
        batch_data = data_generator._padding_batch([instance for instance in reader()])
        cleanup()
        et3 = time.time()
        logger.info("Time to extract features: {}".format(et3 - st3))
        try:
            st4 = time.time()
            probs_split = ds2_model.infer_batch_probs(
                infer_data=batch_data,
                feeding_dict=data_generator.feeding)
            st5 = time.time()
            logger.info("Time to extract probabilities: {}".format(st5 - st4))
            if args.custom_boost:
                probs_split = exp.boost_score(probs_split)

            encoded_prob_scores = [
                codec.encode_sentence_prob(x) for x in probs_split
            ]
            st6 = time.time()
            for scores in encoded_prob_scores:
                logger.info(codec.decode_sentence_prob(scores))
            logger.info("Time to infer: {}, Time to get best guess: {}".format(st5 - st4, st6 - st5))
        except:
            logger.error("Could not compute probability scores")

        if args.decoding_method == "ctc_greedy":
            result_transcripts = ds2_model.decode_batch_greedy(
                    probs_split=probs_split,
                    vocab_list=vocab_list)
        else:
            result_transcripts = ds2_model.decode_batch_beam_search(
                    probs_split=probs_split,
                    beam_alpha=args.alpha,
                    beam_beta=args.beta,
                    beam_size=args.beam_size,
                    cutoff_prob=args.cutoff_prob,
                    cutoff_top_n=args.cutoff_top_n,
                    vocab_list=vocab_list,
                    num_processes=args.num_proc_bsearch)
        result_txts = [codec.trans_to_hindi(txt) for txt in result_transcripts] if args.use_hindi else result_transcripts
        et2 = time.time()

        logger.info("Time to complete inference: {}".format(et2 - st2))
        return encoded_prob_scores, result_txts

    def faster_transcribe(self, file_paths, retry=0):
        if not file_paths:
            return

        if retry < self.max_retries:
            print(retry)
            if not self.worker:
                self.master_conn.close() if self.master_conn is not None else None
                parent_conn, child_conn = Pipe()
                self.worker = Process(target=DeepSpeechTransriber.infer, args=(child_conn,))
                self.master_conn = parent_conn
                self.worker.start()

            self.master_conn.send(file_paths)
            try:
                timeout = 15 if DeepSpeechTransriber.args.use_gpu else 30
                if self.master_conn.poll(timeout):
                    inferred_data = self.master_conn.recv()
                else:
                    raise EOFError("Didn't receive any response for {} seconds".format(timeout))
            except:
                self.logger.exception("Failed to transcribe with {} batch size on retry: "
                                      "{}".format(len(file_paths), retry))
                try:
                    os.kill(int(self.worker.pid), 9)
                    self.kill_child_processes()
                except:
                    self.logger.exception("Failed to kill worker process with pid: {}. "
                        "Maybe its already terminated".format(self.worker.pid))
                self.worker = None

                time.sleep(3)

                # self.transcribe(file_paths, retry + 1)
                num_of_files = len(file_paths)
                output = self.transcribe(file_paths[0:num_of_files/2], retry + 1)
                output.extend(self.transcribe(file_paths[num_of_files/2:], retry + 1))
                return output

            return inferred_data
        else:
            self.logger.error("Reached max retries for {} files: \n{}", len(file_paths), "\n".join(file_paths))

    @staticmethod
    def infer(conn):
        # Prevent logs from Paddle . TODO: Write to a separate file
        from paddle.trainer.config_parser import logger as paddle_logger
        
        exp = ScoringExperiment()
        paddle_logger.setLevel(logging.WARNING)

        logger = logging.getLogger("DeepspeechTranscriber")
        args = DeepSpeechTransriber.args
        logger.setLevel(args.log_level)
        paddle.init(use_gpu=args.use_gpu, rnn_use_batch=True, trainer_count=args.trainer_count)
        st1 = time.time()
        data_generator = DataGenerator(
            vocab_filepath=args.vocab_path,
            mean_std_filepath=args.mean_std_path,
            augmentation_config='{}',
            specgram_type=args.specgram_type,
            keep_transcription_text=False)
        ds2_model = DeepSpeech2Model(
            vocab_size=data_generator.vocab_size,
            num_conv_layers=args.num_conv_layers,
            num_rnn_layers=args.num_rnn_layers,
            rnn_layer_size=args.rnn_layer_size,
            use_gru=args.use_gru,
            pretrained_model_path=args.model_path,
            share_rnn_weights=args.share_rnn_weights)
        ds2_model.logger.setLevel(args.log_level)
        vocab_list = [chars.encode("utf-8") for chars in data_generator.vocab_list]
        print(args.alpha, args.beta, args.lang_model_path,
                                      vocab_list)
        if not args.decoding_method == "ctc_greedy":
            # args.lang_model_path is treated as unicode, so convert to string else
            # swig will complain
            ds2_model.init_ext_scorer(args.alpha, args.beta, str(args.lang_model_path),
                                      vocab_list)
        codec = ProbCodec(vocab_list)
        et1 = time.time()

        logger.info("Time to load model: {}".format(et1 - st1))
        while True:
            file_paths = conn.recv()
            # string is poison pill, normal is list
            if isinstance(file_paths, str):
                break
            st2 = time.time()
            manifest = [{
                "audio_filepath": file_path,
                "text": ""
            } for file_path in file_paths]

            st3 = time.time()
            reader, cleanup = data_generator._instance_reader_creator(manifest)
            batch_data = data_generator._padding_batch([instance for instance in reader()])
            cleanup()
            et3 = time.time()
            logger.info("Time to extract features: {}".format(et3 - st3))

            encoded_prob_scores = []

            try:
                st4 = time.time()
                probs_split = ds2_model.infer_batch_probs(
                    infer_data=batch_data,
                    feeding_dict=data_generator.feeding)
                st5 = time.time()

                if args.custom_boost:
                    probs_split = exp.boost_score(probs_split)

                encoded_prob_scores = [
                    codec.encode_sentence_prob(x) for x in probs_split
                ]
                st6 = time.time()
                for scores in encoded_prob_scores:
                    logger.info(codec.get_best_guess(scores))
                logger.info("Time to infer: {}, Time to get best guess: {}".format(st5 - st4, st6 - st5))
                logger.info(codec.get_best_guess(";".join(encoded_prob_scores)))
            except:
                logger.error("Could not compute probability scores")

            if args.decoding_method == "ctc_greedy":
                result_transcripts = ds2_model.decode_batch_greedy(
                        probs_split=probs_split,
                        vocab_list=vocab_list)
            else:
                result_transcripts = ds2_model.decode_batch_beam_search(
                        probs_split=probs_split,
                        beam_alpha=args.alpha,
                        beam_beta=args.beta,
                        beam_size=args.beam_size,
                        cutoff_prob=args.cutoff_prob,
                        cutoff_top_n=args.cutoff_top_n,
                        vocab_list=vocab_list,
                        num_processes=args.num_proc_bsearch)
            result_txts = [codec.trans_to_hindi(txt) for txt in result_transcripts] if args.use_hindi else result_transcripts
            et2 = time.time()

            logger.info("Time to complete inference: {}".format(et2 - st2))
            conn.send(zip(encoded_prob_scores, result_txts))

    def kill_child_processes(self, term_signal=9):
        parent_pid = os.getpid()
        # OS Specific since underlying command doesn't work on OSX or Windows
        if platform == "linux" or platform == "linux2":
            ps_command = subprocess.Popen("ps -o pid --ppid %d --noheaders" % parent_pid, shell=True, stdout=subprocess.PIPE)
            ps_output = ps_command.stdout.read()
            retcode = ps_command.wait()
            assert retcode == 0, "ps command returned %d" % retcode
            for pid_str in ps_output.split("\n")[:-1]:
                try:
                    os.kill(int(pid_str), term_signal)
                except:
                    self.logger.info("Could not kill process {}".format(pid_str))



if __name__ == '__main__':
    transcriber = DeepSpeechTransriber()
    audio_file = DeepSpeechTransriber.args.audio_file
    DeepSpeechTransriber.args.log_level = logging.ERROR
    [(confidence, content)] = transcriber.transcribe([audio_file])
    print (content)
    print (confidence)
    os.kill(int(transcriber.worker.pid), 9)
    sys.exit()
