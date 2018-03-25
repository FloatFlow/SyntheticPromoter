"""
Written by Wolfgang Rahfeldt
For Synthetic Project project, Michael Jensen Lab
Advised by Jia Wei, Rithun M. 
Last Edited 17 September 2017

This is a simple multiprocessing script to run on local processors.
Given DNA sequencing data and position weight matrices, it will analyze the sequences for transcription factor binding sites. 

"""

#import libraries
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.stats as stats
from scipy.stats import binom
import scipy
import itertools
from collections import Counter
from collections import defaultdict
import functools
import Bio
from Bio import motifs
import multiprocessing as mp
import tqdm
import argparse

#available arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s', 
                    '--seqfile', 
                    type=str, 
                    default='C:/Users/Wolf/Desktop/SynPro/Complete Raw data for sorted cells/S1.fa.uniq.tcga.all.csv',
                    help='CSV file with sequencing data. Single column with counts and DNA sequence seperated by a single space.')
parser.add_argument('-j', 
                    '--jaspar', 
                    type=str, 
                    default='C:/Users/Wolf/Documents/BioScripts/Jaspar_lib.txt',
                    help= 'JASPAR txt file that contains all the position weight matrices for the TREs you want to search for.')
parser.add_argument('-o', 
                    '--output', 
                    type=str, 
                    default='c:/users/wolf/desktop/p1_seqtres.csv',
                    help= 'Full path (directory and filename) for output CSV.')
parser.add_argument('-c', 
                    '--count', 
                    type=int, 
                    default=5, 
                    help= 'Minimum count for a DNA sequence from DNAseq. Used for quality control. Default is 5.')
parser.add_argument('-t', 
                    '--threshold',
                    type=float, 
                    default=10.0,
                    help= 'Minimum log-likelihood for a TRE-sequence association to be kept. Default is 10.')

#if len(sys.argv)==1:
#    parser.print_help()
#    sys.exit(1)
args = parser.parse_args()

#read in a csv with a single column, split to divide '# of hits' from nucleotide sequence, filter out TREs with < 5 hits
def read_in(filepath):
    dfs1 = pd.read_csv(filepath, engine='python', sep=None, header=None)
    dfs1.columns = ['count','sequence']
    dfs1 = dfs1.loc[dfs1['count'] > args.count]
    return dfs1

#get sequences
first_df = read_in(args.seqfile)
ana_seqs = first_df.sequence.values
#first_df['sequence'].replace(regex=True,inplace=True,to_replace=r'TCGAGTAGAGTCTAGACTCTACATTTTGACACCCCCA',value=r'')

#Collect all TRE names, in order
with open(args.jaspar, 'r') as f:
    name_lst = []
    for ln in f:
        if ln.startswith('>'):
            name_lst.append(ln[10:-1])

#get frequency matrix for each TRE
fh = open(args.jaspar)
pssms = [m.counts.normalize(pseudocounts=0.25).log_odds() for m in motifs.parse(fh, "jaspar")]

#zip together TRE names and their respective matrix into dictionary
super_matrix = dict(zip(name_lst, pssms))
#super_matrix = dict(zip(range(len(name_lst)), pssms)) #this might be slightly faster, but more annoying later on


def seq_analyzer(x):  
    trehits = []
    for key, value in super_matrix.items():
        for position, score in value.search(Bio.Seq.Seq(ana_seqs[x], value.alphabet), threshold=args.threshold):
            trehits.append([x, key, position, score])
    return trehits

def main():
    pool = mp.Pool(processes=3)
    results = pool.imap(seq_analyzer, range(0,len(ana_seqs)), 
                        chunksize = int(len(ana_seqs)/3))

    for _ in tqdm.tqdm(pool.imap(seq_analyzer, range(0,len(ana_seqs))),
                       total=len(range(0,len(ana_seqs)))):
        pass

    tre_df = pd.DataFrame([item for sublist in results for item in sublist], 
                          columns = ['seq_num', 'tre', 'position', 'logprob'])
    tre_df.to_csv(args.output, index=False)

if __name__ == '__main__':
    main()