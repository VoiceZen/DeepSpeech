"""Build vocabulary from manifest files.

Each item in vocabulary file is a character.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import codecs
import json
from collections import Counter
import os.path
import _init_paths
from data_utils.utility import read_manifest
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('count_threshold',  int,    0,  "Truncation threshold for char counts.")
add_arg('vocab_path',       str,
        'data/librispeech/vocab.txt',
        "Filepath to write the vocabulary.")
add_arg('type',       str,
        'csv',
        "file type of manifest.")

add_arg('manifest_paths',   str,
        None,
        "Filepaths of manifests for building vocabulary. "
        "You can provide multiple manifest files.",
        nargs='+',
        required=True)
# yapf: disable
args = parser.parse_args()


def count_manifest(counter, manifest_path,file_type):
    manifests = read_manifest(manifest_path, type=file_type)
    for line in manifests:
        try:
            for char in line['text']:
                counter.update(char)
        except:
            import pdb; pdb.set_trace()


def main():
    print_arguments(args)

    counter = Counter()
    for manifest_path in args.manifest_paths:
        count_manifest(counter, manifest_path,args.type)

    count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    with codecs.open(args.vocab_path, 'w', 'utf-8') as fout:
        for char, count in count_sorted:
            if count < args.count_threshold: break
            try:
                fout.write(char + '\n')
            except:
                import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
