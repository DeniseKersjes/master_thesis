#!/usr/bin/env python
"""
Author: Denise Kersjes (student number 950218-429-030)
Date: 9 January 2018
Script for converting features of a specific sequence into correct input for convolutional neural network.

Output is a numpy array containing feature scores of a particular sequence.
"""

import numpy as np
from optparse import OptionParser

class inputCNN():
    """ This object is designed to keep track of the different data files containing features of a specific sequence.
    """

    def __init__(self, data_directory):
        """ Initialize numpy array with features scores for a particular sequence.

        data_directory: string, working directory where the data can be found

        Defined parameters:
        self.genome: open TSV file, containing sequences of particular nucleotide positions of the Human genome
        self.phastcon_mam: open TSV file,
        self.phastcon_pri: open TSV file,
        self.phastcon_verp: open TSV file,
        self.phylop_mam: open TSV file,
        self.phylop_pri: open TSV file,
        self.phylop_verp: open TSV file,
        self.gerp: open TSV file,
        self.gerp_elem: open TSV file,
        """

        # Convert the working directory to an object variable
        self.data_directory = data_directory

        # Define the different data files with features as an object variable
        self.genome = open(data_directory + "Genome_out.tsv", "r")
        self.phastcon_mam = open(data_directory + "mamPhastcon_out.tsv", "r")
        self.phastcon_pri = open(data_directory + "priPhastcon_out.tsv", "r")
        self.phastcon_verp = open(data_directory + "verpPhastcon_out.tsv", "r")
        self.phylop_mam = open(data_directory + "mamPhylop_out.tsv", "r")
        self.phylop_pri = open(data_directory + "priPhylop_out.tsv", "r")
        self.phylop_verp = open(data_directory + "verpPhylop_out.tsv", "r")
        self.gerp = open(data_directory + "Gerp_out.tsv", "r")
        self.gerp_elem = open(data_directory + "GerpElem_out.tsv", "r")

    def nt_features(self, file):
        """ Return: list per nucleotide containing boolean numbers to indicate how a particular sequence looks like

        file: open TSV file, containing sequences of particular nucleotide positions of the Human genome

        Defined variables:
        ntA: list of lists, containing boolean numbers to indicate where adenine's are located in a particular sequence
        ntC: list of lists, containing boolean numbers to indicate where cytosine's are located in a particular sequence
        ntT: list of lists, containing boolean numbers to indicate where thymine's are located in a particular sequence
        ntG: list of lists, containing boolean numbers to indicate where guanine's are located in a particular sequence
        """

        # Define a list per nucleotide
        ntA = []
        ntC = []
        ntT = []
        ntG = []

        for line in file:
            header = line.startswith('#')
            # The header line should not be considered for data processing
            if header == False:
                # The genome file is tab delimited, and the sequence is located after the second tab
                seq = line.split("\t")[2]
                # Define new nucleotides list per sequence
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
                # Extend the objected nucleotide list with the new sequence
                ntA.append(A)
                ntC.append(C)
                ntT.append(T)
                ntG.append(G)

        return ntA, ntC, ntT, ntG

    def phastcon_phylop_scores(self, file):
        """ Return: list of lists containing PhastCon or PhyloP scores per sequence

        """

        # PhastCon files are tab delimited, and the scores are located after the second tab
        # The scores are comma separated
        parsed_file = list(map(lambda line: (line.strip('\n').split('\t')[2]).split(','), file))

        # The first list of the parsed file list contains the header line, so this list is excluded
        return parsed_file[1:]

    def gerp_scores(self, file):
        """
        :param file:
        :return:
        """

        # Gerp files are tab delimited,and the scores are located after the second tab
        # The scores are comma separated
        parsed_file = list(map(lambda line: (line.strip('\n').split('\t')[2]).split(','), file))

        # Every even number (start counting with 0) in the Gerp score list of a sequence is the GerpN or GerpRS score
        # Every odd number of the score list is the GerpS score or the p-value of GerpRS
        Gerp1 = [] # Can be GerpN or GerpRS
        Gerp2 = [] # Can be GerpS or Gerp_pval
        for score_list in parsed_file[1:]:
            # Create temporarily lists for every sequence
            Gerp1_tmp = []
            Gerp2_tmp = []
            for index, score in enumerate(score_list):
                # Convert the not available scores to NoneType
                if score == '-':
                    score = None
                if index % 2 == 0:
                    Gerp1_tmp.append(score)
                else:
                    Gerp2_tmp.append(score)
            # Extend the Gerp lists with the temporary ones
            Gerp1.append(Gerp1_tmp)
            Gerp2.append(Gerp2_tmp)

        return Gerp1, Gerp2

    def combine(self):
        """ Return numpy array to store the different features scores of a particular sequence.

        Defined variables:
        ntA: list of lists, containing boolean numbers to indicate where adenine's are located in a particular sequence
        ntC: list of lists, containing boolean numbers to indicate where cytosine's are located in a particular sequence
        ntT: list of lists, containing boolean numbers to indicate where thymine's are located in a particular sequence
        ntG: list of lists, containing boolean numbers to indicate where guanine's are located in a particular sequence
        phastcon_mam: list of lists, containing floats which represent mammalian PhastCon scores
        phastcon_pri: list of lists, containing floats which represent primate PhastCon scores
        phastcon_verp list of lists, containing floats which represent vertebrate PhastCon scores
        phylop_mam: list of lists, containing floats which represent mammalian PhyloP scores
        phylop_pri: list of lists, containing floats which represent primate PhyloP scores
        phylop_verp list of lists, containing floats which represent vertebrate PhyloP scores
        GerpN: list of lists, containing floats (or NoneTypes) which represent the neutral evolution scores by GERP++
        GerpS: list of lists, containing floats (or NoneTypes) which represent rejected substitution scores by GERP++
        GerpRS: list of lists, containing floats (or NoneTypes) which represent the Gerp element scores
        Gerp_pval: list of lists, containing floats (or NoneTypes) which represent Gerp element p-values
        """

        # Get the nucleotides scores
        ntA, ntC, ntT, ntG = inputCNN(data_directory).nt_features(self.genome)

        # Get the PhastCon scores and the PhyloP scores
        phastcon_mam = inputCNN(data_directory).phastcon_phylop_scores(self.phastcon_mam)
        phastcon_pri = inputCNN(data_directory).phastcon_phylop_scores(self.phastcon_pri)
        phastcon_verp = inputCNN(data_directory).phastcon_phylop_scores(self.phastcon_verp)
        phylop_mam = inputCNN(data_directory).phastcon_phylop_scores(self.phylop_mam)
        phylop_pri = inputCNN(data_directory).phastcon_phylop_scores(self.phylop_pri)
        phylop_verp = inputCNN(data_directory).phastcon_phylop_scores(self.phylop_verp)
        # Get the Gerp scores
        GerpN, GerpS = inputCNN(data_directory).gerp_scores(self.gerp)
        GerpRS, Gerp_pval = inputCNN(data_directory).gerp_scores(self.gerp_elem)

        zipped = list(zip(ntA, ntC, ntT, ntG,
                          phastcon_mam, phastcon_pri, phastcon_verp,
                          phylop_mam, phylop_pri, phylop_verp,
                          GerpN, GerpS, GerpRS, Gerp_pval))

        input_format = np.array(zipped, dtype='float')

        print(input_format)
        # return input_format


if __name__ == "__main__":

    # Specify the options for running from the command line
    parser = OptionParser()
    # Specify the data directory
    parser.add_option("-p", "--path", dest="path", help="Path to the output of the 'find_sequence.py' script that\
     finds for genomic locations the associated sequences and annotations within a defined window size upstream and\
      downstream of the query location.", default="")

    # Get the command line options
    (options, args) = parser.parse_args()
    data_directory = options.path

    # Run the main function
    inputCNN(data_directory).combine()

    # "/mnt/scratch/kersj001/data/output/test/test_ben2/"
