'''
This script converts sequences in a dataframe into numpy arrays, where sequential position of a character is converted into an x, y position in a 2d grid. 
The 3rd dimension corresponds to one-hot coding for each nucleotide (4)
'''


import pandas as pd
import random
import itertools
import numpy as np
from tqdm import *
import multiprocessing as mp
import os
from sklearn.model_selection import train_test_split

def _binary_repr(num, width):
    """Return a binary string representation of `num` zero padded to `width`
    bits."""
    return format(num, 'b').zfill(width)

def _hilbert_integer_to_transpose(h, p, N):
    """Store a hilbert integer (`h`) as its transpose (`x`).
    :param h: integer distance along hilbert curve
    :type h: ``int``
    :param p: number of iterations in Hilbert curve
    :type p: ``int``
    :param N: number of dimensions
    :type N: ``int``
    """
    h_bit_str = _binary_repr(h, p*N)
    x = [int(h_bit_str[i::N], 2) for i in range(N)]
    return x

def _transpose_to_hilbert_integer(x, p, N):
    """Restore a hilbert integer (`h`) from its transpose (`x`).
    :param x: the transpose of a hilbert integer (N components of length p)
    :type x: ``list`` of ``int``
    :param p: number of iterations in hilbert curve
    :type p: ``int``
    :param N: number of dimensions
    :type N: ``int``
    """
    x_bit_str = [_binary_repr(x[i], p) for i in range(N)]
    h = int(''.join([y[i] for i in range(p) for y in x_bit_str]), 2)
    return h

def coordinates_from_distance(h, p, N):
    """Return the coordinates for a given hilbert distance.
    :param h: integer distance along the curve
    :type h: ``int``
    :param p: side length of hypercube is 2^p
    :type p: ``int``
    :param N: number of dimensions
    :type N: ``int``
    """
    x = _hilbert_integer_to_transpose(h, p, N)
    Z = 2 << (p-1)

    # Gray decode by H ^ (H/2)
    t = x[N-1] >> 1
    for i in range(N-1, 0, -1):
        x[i] ^= x[i-1]
    x[0] ^= t

    # Undo excess work
    Q = 2
    while Q != Z:
        P = Q - 1
        for i in range(N-1, -1, -1):
            if x[i] & Q:
                # invert
                x[0] ^= P
            else:
                # excchange
                t = (x[0] ^ x[i]) & P
                x[0] ^= t
                x[i] ^= t
        Q <<= 1

    # done
    return x

def distance_from_coordinates(x, p, N):
    """Return the hilbert distance for a given set of coordinates.
    :param x: coordinates len(x) = N
    :type x: ``list`` of ``int``
    :param p: side length of hypercube is 2^p
    :type p: ``int``
    :param N: number of dimensions
    :type N: ``int``
    """
    M = 1 << (p - 1)

    # Inverse undo excess work
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(N):
            if x[i] & Q:
                x[0] ^= P
            else:
                t = (x[0] ^ x[i]) & P
                x[0] ^= t
                x[i] ^= t
        Q >>= 1

    # Gray encode
    for i in range(1, N):
        x[i] ^= x[i-1]
    t = 0
    Q = M
    while Q > 1:
        if x[N-1] & Q:
            t ^= Q - 1
        Q >>= 1
    for i in range(N):
        x[i] ^= t

    h = _transpose_to_hilbert_integer(x, p, N)
    return h


# helper functions

# in case you need to make some mock data
def seq_generator(n_seqs, seq_len):
    seq_list = []
    for i in range(n_seqs):
        seq_list.append(''.join(random.choice(['A', 'C', 'G', 'T']) for _ in range(seq_len)))
    return seq_list

#mock_seqs = seq_generator(20, 20)

# split sequences into a list of kmers
def kmer_sieve(seq, mer_size):
    #out_list = []
    #for seq in seq_list:
    #    seq_mers = [seq[i:i+mer_size] for i in range(len(seq)-mer_size+1)]
    #    #for i in range(len(seq_list)-mer_size+1):
    #    #    seq_mers.append(seq[i:i+mer_size])
    #    out_list.append(seq_mers)
    mer_list = [seq[i:i+mer_size] for i in range(len(seq)-mer_size+1)]
    return mer_list


# all possible permutations of a certain kmer size
kmer_perms = list(''.join(kmer) for kmer in itertools.product(['A', 'C', 'G', 'T'], repeat=1)) #was originally in kmer_onehotcoder, with repeat=mer_size


#initialize one-hot vectors
z_array = np.zeros((len(kmer_perms), len(kmer_perms)))
insert_list = [1]*len(kmer_perms)
inxs = np.diag_indices_from(z_array)
z_array[inxs] = insert_list

#map vectors to kmers
hot_dict = dict(zip(kmer_perms, z_array))


# one-hot encode kmers generated from given sequences
def kmer_onehotcoder(seq, mer_size, hot_dict):

    #vectorize sequences
    vectorized_seq = kmer_sieve(seq, mer_size)
    #for seq in vectorized_seq:
    for i, e in enumerate(vectorized_seq):
        if e in hot_dict:
            vectorized_seq[i] = hot_dict[e]
    return vectorized_seq

def hilbert_image_generator(kmerized_array):
    
    # determine hilbert power appropriate for length of sequence
    hilbert_power = 5
    #while len(kmerized_array) > (2**hilbert_power)**2: #to solve for it every time
    #    hilbert_power += 1
    
    # initialize appropriately sized array
    init_array = np.zeros((2**hilbert_power,2**hilbert_power, len(kmerized_array[0])))
    
    # replace default array at certain "pixel" location with vectorized kmer
    #crop_row = 0  #dynamicly determine appropriate horizontal drop rows
    for i in range(len(kmerized_array)):
        pix_position = coordinates_from_distance(i, hilbert_power, 2)
        init_array[pix_position[0], pix_position[1]] = kmerized_array[i]
        #if pix_position[0] + 1 > crop_row:
        #    crop_row = pix_position[0] + 1
        #print(coordinates_from_distance(i, hilbert_power, 2))
        #print(kmerized_array[i])
    
    #return array, horizontally sliced to exclude empty portion of array
    #return init_array[:22,:,:]
    return init_array

data = pd.read_csv('D:/Projects/iSynPro/iSynPro/HilbertCNN/train_val_npys/test_df.csv', index_col=0).reset_index()

#print(max([len(seq) for seq in data['sequence']]))

write_folder = 'D:/Projects/iSynPro/iSynPro/HilbertCNN/train_val_npys/1mer/test'
def multiprocessing_write_wrapper(i):
    # create and write hilbert curves
    hilbert_curve = hilbert_image_generator(kmer_onehotcoder(data['sequence'][i], 1, hot_dict))
    if data['y'][i] == 0:
        save_path = '{}/low/{}.npy'.format(write_folder, i) 
        np.save(save_path, hilbert_curve)
    if data['y'][i] == 1:
        save_path = '{}/high/{}.npy'.format(write_folder, i) 
        np.save(save_path, hilbert_curve)


def main():
    #set up multiprocessing for (total processors - 1)
    pool = mp.Pool(processes=3)

    with tqdm(total = data.shape[0]) as pbar:
        for f in tqdm(pool.imap(multiprocessing_write_wrapper, range(data.shape[0]))):
            pbar.update()

if __name__ == '__main__':
    main()
