import os
import uuid
import itertools
import subprocess
import numpy as np

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class ScoringExperiment(object):
    def boost_score(self, prob_splits):
        # prob_splits is a list of numpy.ndarray
        for prob_split in prob_splits:
            # prob_split is a ndarray
            ctc_pos = prob_split.shape[1] - 1
            for i, x in enumerate(prob_split):
                positions = np.argwhere(x > 0.01).transpose()[0]
                # if ctc probability is over threshold, distribute among others
                if x[ctc_pos] > 0.5:
                    if len(positions) > 1:
                        delta = x[ctc_pos]/len(positions)
                        x[ctc_pos] = delta
                        for pos in positions:
                            if not pos == ctc_pos:
                                x[pos] = x[pos] + delta
        return prob_splits


class ProbCodec(object):
    """Encoding logits and decoding to logits."""

    """
    Encodes and Decodes probability of each character in a utterance
    encoded value looks like ~9|3-k8n2|3-~9-~a|30 i.e each frame is separated
    by "-", in the frame one or more characters could be mentioned k8n2 means
    k has probability of 0.8 and n of 0./2 the encoding is RLE, so ~a|30 means
    ~ has a probability of 10 which continues for 30 frames. '~' is used as
    frame delimiter, space is more suitable, but deep speech vocab uses it
    for frame pause
    """

    # custom delimeter for run length encoding
    rle_delimiter = "|"

    def __init__(self, vocab_dict):
        self._vocab_dict = vocab_dict
        vocab_size = len(vocab_dict)
        self._vocab_dict[vocab_size] = "~"
        self._reverse_vocab_dict = {v: k for k, v in self._vocab_dict.items()}

    @staticmethod
    def softmax(x):
        m = np.expand_dims(np.max(x, axis=-1), -1)
        e = np.exp(x - m)
        return e / np.expand_dims(e.sum(axis=-1), -1)

    def _get_character(self, position):
        """~ represents word break."""
        return self._vocab_dict[position]

    def _encode_prob_score(self, probability):
        """Converts softmax probability to hex scale of 1-10 i.e 10 is a."""
        enc_prob = ''
        enc_int = int(round(probability, 1) * 10)
        if enc_int == 10:
            enc_prob = 'a'
        else:
            enc_prob = str(enc_int)

        return enc_prob

    def _encode_char_prob(self, prob_split):
        """
        For vocabs emits an encoded value e.g k8n2 means.

        k has a score of 8 and n a score of 2
        """
        row_encoding = ""
        for index, x in enumerate(prob_split):
            if x > 0.05:
                token = self._get_character(index) + self._encode_prob_score(x)
                row_encoding += token

        return row_encoding

    def _get_rle_value(self, token, repeat_count):
        if token == "":
            return ""
        elif repeat_count == 1:
            return token
        return token + ProbCodec.rle_delimiter + str(repeat_count)

    def _decode_normal(self, row):
        # create row of 0 values for vocab_dict length,
        # set value for right position
        decoded_row = [0] * len(self._vocab_dict)
        for i in range(0, len(row), 2):
            target_char = row[i:i + 1]
            prob_score = row[i + 1:i + 2]
            try:
                x = self._reverse_vocab_dict[target_char]
                decoded_row[x] = int(prob_score, 16)
            except Exception as e:
                print(target_char)
                raise e

        return decoded_row

    def _decode_rle(self, row):
        decoded_values = []
        root_token, repeats = self._get_repeat_count(row)
        decoded_value = self._decode_normal(root_token)
        for _ in itertools.repeat(None, repeats):
            decoded_values.append(decoded_value)
        return decoded_values

    def _get_repeat_count(self, row):
        repeat_index = row.find(ProbCodec.rle_delimiter)
        if repeat_index > 0:
            return row[0:repeat_index], int(row[repeat_index + 1:])
        return row, 1

    def encode_sentence_prob(self, jasper_logits):
        logits = self.softmax(jasper_logits)

        enc_sen_prob = ''
        repeat_count = 1
        last_token = ""
        for x in logits:
            token = self._encode_char_prob(x)
            # if CTC
            if token == last_token:
                repeat_count = repeat_count + 1
            else:
                rle_token = self._get_rle_value(last_token, repeat_count)
                repeat_count = 1
                last_token = token
                if rle_token != "":
                    enc_sen_prob = enc_sen_prob + rle_token + "-"

        return enc_sen_prob + self._get_rle_value(last_token, repeat_count)

    def decode_sentence_prob(self, enc_str):
        decoded_sentence = []
        rows = enc_str.split("-")
        for row in rows:
            decoded_rows = self._decode_rle(row)
            decoded_sentence.extend(decoded_rows)
        return np.array(decoded_sentence)

def convert_audio(audio_object):
    '''Convert the audio buffer to static audio'''
    wav_name = str(uuid.uuid4())
    src_directory = "/data/vz/services/raw"
    source = os.path.join(src_directory, wav_name+".wav")
    dest_directory = "/data/vz/services/processed"
    if not os.path.exists(src_directory):
        os.makedirs(src_directory)
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    dest = os.path.join(dest_directory, wav_name+".wav")
    audio_object.save(source)
    command = "ffmpeg -y -i  "+source+" -acodec pcm_s16le -ar 8000 -ac 1 "+dest
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()

    if process.returncode != 0:
        print("Command failed "+command)
        return None

    return dest
