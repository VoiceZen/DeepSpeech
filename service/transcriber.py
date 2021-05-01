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
    args = {}

    def __init__(self, asr_config, max_retries = 5):
        DeepSpeechTransriber.args = asr_config
        arg = asr_config
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

    def old_transcribe(self, file_paths):
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
        print(batch_data)
        st4 = time.time()
        print('using Gpu',args.use_gpu)
        probs_split = ds2_model.infer_batch_probs(
            infer_data=batch_data,
            feeding_dict=data_generator.feeding)
        st5 = time.time()
        logger.info("Time to extract probabilities: {}".format(st5 - st4))
        try:
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

    def transcribe(self, file_paths, retry=0):
        if not file_paths:
            return

        if retry < self.max_retries:
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
        exp = ScoringExperiment()
        paddle_logger.setLevel(logging.WARNING)
        
        logger = logging.getLogger("DeepspeechTranscriber")
        args = DeepSpeechTransriber.args
        logger.setLevel(args.log_level)
        logger.info("GPU FLAG: %s" % args.use_gpu)
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
        codec = ProbCodec({itr: value for itr, value in enumerate(vocab_list)})
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

            logger.error("Time to extract features: {}".format(et3 - st3))
            encoded_prob_scores = []

            try:
                st4 = time.time()
                probs_split = ds2_model.infer_batch_probs(
                    infer_data=batch_data,
                    feeding_dict=data_generator.feeding)
                st5 = time.time()

                logger.error(probs_split)

                if args.custom_boost:
                    probs_split = exp.boost_score(probs_split)

                encoded_prob_scores = [
                    codec.encode_sentence_prob(x) for x in probs_split
                ]
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
            logger.info(result_txts)
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