# coding=utf-8
from __future__ import print_function

import sys

import torch
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

from model.reconstruction_model import Reconstructor
from asdl.lang.py.dataset import Django
from components.dataset import Dataset, Example

def reconstructor_augmentation(args, labeled_examples, unlabeled_examples, model):


    all_unlabel_code = [e.tgt_code for e in unlabeled_examples]
    all_unlabel_gold_nl = [e.src_sent for e in unlabeled_examples]
    all_unlabel_hyp_nl = []

    # decode code to nl utterance
    for code in tqdm(all_unlabel_code):
        nl_unlabel_hyps = model.sample(code)
        all_unlabel_hyp_nl.append(nl_unlabel_hyps[0])

    bleu_score = corpus_bleu([[ref] for ref in all_unlabel_gold_nl], all_unlabel_hyp_nl)
    print('reconstructed BLEU %.2f' % (bleu_score*100.0), file=sys.stderr)

    # get the examples for the augmented dataset
    all_unlabel_nl = [' '.join(e) for e in all_unlabel_hyp_nl]
    (augmented_unlabel_examples, dev, test), vocab = Django.parse_django_dataset(all_unlabel_nl, all_unlabel_code,
                                                            'asdl/lang/py/py_asdl.txt', vocab_freq_cutoff=5, 
                                                            verbose=False, direct=True, ignore_code=True)
    assert len(dev)==len(test)==0

    # derive the final train set
    train_set = labeled_examples + augmented_unlabel_examples

    return train_set, vocab
