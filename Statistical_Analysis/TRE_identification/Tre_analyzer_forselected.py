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


jia_location = 'c:/users/wolf/desktop/SynPro/Tre_analysis_tandem_promoters.csv'
first_df = pd.read_csv(jia_location)
#first_df['sequence'].replace(regex=True,inplace=True,to_replace=r'TCGAGTAGAGTCTAGACTCTACATTTTGACACCCCCA',value=r'')

#location of Jaspar library
libaddress = 'C:/Users/Wolf/Documents/BioScripts/Jaspar_lib.txt'
fh = open(libaddress)

#Collect all TRE names, in order
with open(libaddress, 'r') as f:
    name_lst = []
    for ln in f:
        if ln.startswith('>'):
            name_lst.append(ln[10:-1])

#get frequency matrix for each TRE
pssms = [m.counts.normalize(pseudocounts=0.25).log_odds() for m in motifs.parse(fh, "jaspar")]

#zip together TRE names and their respective matrix into dictionary
super_matrix = dict(zip(name_lst, pssms))
#super_matrix = dict(zip(range(len(name_lst)), pssms))  #use this one for faster performance

ana_seqs = first_df.Sequence.values

def seq_analyzer(x):  
    trehits = []
    for key, value in super_matrix.items():
        for position, score in value.search(Bio.Seq.Seq(ana_seqs[x], value.alphabet), threshold=10.0):
            trehits.append([x, key, position, score])
    return trehits


if __name__ == '__main__':
    pool = mp.Pool(processes=3)
    results = pool.imap(seq_analyzer, range(0,len(ana_seqs)), chunksize = int(len(ana_seqs)/3))
    for _ in tqdm.tqdm(pool.imap(seq_analyzer, range(0,len(ana_seqs))), total=len(range(0,len(ana_seqs)))):
        pass
    colnames = ['seq_num', 'tre', 'position', 'logprob']
    tre_df = pd.DataFrame([item for sublist in results for item in sublist], columns = colnames)
    destination = 'c:/users/wolf/desktop/SynPro/tandemseqs_tres.csv'
    tre_df.to_csv(destination, index=False)