#!/usr/bin/env python
"""
Author: Denise Kersjes (student number 950218-429-030)
Script for converting features of a specific sequence into correct input for convolutional neural network.

Output is a numpy array containing feature scores of a particular sequence.
"""

import numpy as np
import tensorflow as tf


class inputCNN():
    """ Numpy array to store the different features scores of a particular sequence.

    This object is designed to keep track of the different data files containing features of specific sequence.
    """

    def __init__(self, data_directory):
        """ Initialize numpy array with features scores for a particular sequence.

        data_directory: string, working directory where the data can be found

        defined parameters

        """

        # convert the working directory to an object variable
        self.data_directory = data_directory

        # define the different data files with features as an object variable
        self.genome = open(data_directory + "Genome_out.tsv", "r")
        self.phastcon_mam = open(data_directory + "mamPhastcon_out.tsv", "r")
        self.phastcon_pri = open(data_directory + "priPhastcon_out.tsv", "r")
        self.phastcon_verp = open(data_directory + "verpPhastcon_out.tsv", "r")
        self.phylop_mam = open(data_directory + "mamPhylop_out.tsv", "r")
        self.phylop_pri = open(data_directory + "priPhylop_out.tsv", "r")
        self.phylop_verp = open(data_directory + "verpPhylop_out.tsv", "r")
        self.gerp = open(data_directory + "Gerp_out.tsv", "r")
        self.elem = open(data_directory + "GerpElem_out.tsv", "r")

        self.file = open(data_directory + "input_CNN.txt", "w")
        # list(map(lambda line: self.file.write(line.split('\t')[2]), self.genome))

    def nt_features(self):
        """ Return: list per nucleotide containing boolean numbers for the particular nucleotide

        """

        # define a list per nucleotide
        ntA = []
        ntC = []
        ntT = []
        ntG = []

        for line in self.genome:
            header = line.startswith('#')
            # the header line should not be considered for data processing
            if header == False:
                # the genome file is tab delimited, and the sequence is located after the second tab
                seq = line.split("\t")[2]
                # define new nucleotides list per sequence
                A = []
                C = []
                T = []
                G = []
                for nt in (seq):
                    if nt == 'A':
                        A.append(1)
                        C.append(0)
                        T.append(0)
                        G.append(0)
                    if nt == 'C':
                        A.append(0)
                        C.append(1)
                        T.append(0)
                        G.append(0)
                    if nt == 'T':
                        A.append(0)
                        C.append(0)
                        T.append(1)
                        G.append(0)
                    if nt == 'G':
                        A.append(0)
                        C.append(0)
                        T.append(0)
                        G.append(1)
                # extend the objected nucleotide list with the new sequence
                ntA.append(A)
                ntC.append(C)
                ntT.append(T)
                ntG.append(G)

        return ntA, ntC, ntT, ntG

    def phastcon(self):
        """ Return: PhastCon scores per sequence

        """

        #
        #parsing = lambda line: [(line.strip('\n').split('\t')[2]).split(',')] if (line.startswith('#') == False) else []

        mam = list(map(lambda line: [(line.strip('\n').split('\t')[2])] , self.phastcon_mam))
        print(mam[1:])
        mam2 = list(map(lambda scores: scores.split(','), mam[1:]))
        print(mam2)
        # for i in self.phastcon_mam:
        #     print(i)

    def combine(self):
        """

        :return:
        """

        # get the nucleotides scores
        ntA, ntC, ntT, ntG = inputCNN(data_directory).nt_features()

        print(ntA)
        # get the PhastCon scores
        inputCNN(data_directory).phastcon()



        zipped = list(zip(ntA, ntC, ntT, ntG))
        # for i in zipped:
        #     print(i)
        # print(zipped)

        ty = np.array(zipped)
        print(ty)

        # list(map(lambda seq: self.file.write(str(seq) + '\n'), ntA))

if __name__ == "__main__":
    """
    """
    data_directory = "/mnt/scratch/kersj001/data/output/test/test_ben_none/"
    # inputCNN(data_directory)
    inputCNN(data_directory).combine()