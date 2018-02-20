#!/usr/bin/env python
"""
Author: Denise Kersjes (student number 950218-429-030)
Date: 31 January 2018
Script for converting features of a specific sequence into correct input for convolutional neural networks

Output is a H5py file with the compressed numpy array containing feature scores of a particular sequence
"""

import time
import numpy as np
import h5py
import pdb
from optparse import OptionParser


class inputCNN():
    """ This object is designed to keep track of the different data files containing features of a specific sequence
    """

    def __init__(self, data_directory):
        """ Initialize numpy array with features scores for a particular sequence.

        data_directory: string, working directory where the data can be found

        Defined parameters:
        self.genome: open TSV file, containing sequences of particular nucleotide positions of the Human genome
        self.phastcon_mam: open TSV file, containing mammalian PhastCon scores of particular nucleotide positions
        self.phastcon_pri: open TSV file, containing primate PhastCon scores of particular nucleotide positions
        self.phastcon_verp: open TSV file, containing vertebrate PhastCon scores of particular nucleotide positions
        self.phylop_mam: open TSV file, containing mammalian PhyloP scores of particular nucleotide positions
        self.phylop_pri: open TSV file, containing primate PhyloP scores of particular nucleotide positions
        self.phylop_verp: open TSV file, containing vertebrate PhyloP scores of particular nucleotide positions
        self.gerp: open TSV file, containing neutral evolution scores (GerpN) and rejected substitution scores (GerpS)\
         of particular nucleotide positions
        self.gerp_elem: open TSV file, containing Gerp element scores (GerpRS) and p-values (Gerppval) of particular\
         nucleotide positions
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
                    if nt == 'N':
                        print("WARNING: Sequence contains 'N' values")
                        A.append(0)
                        C.append(0)
                        T.append(0)
                        G.append(0)
                # Extend the objected nucleotide list with the new sequence
                ntA.append(A)
                ntC.append(C)
                ntT.append(T)
                ntG.append(G)

        return ntA, ntC, ntT, ntG

    def phastcon_phylop_scores(self, file):
        """ Return: list of lists containing per sequence PhastCon or PhyloP scores in Strings or NoneTypes

        file: open TSV file, containing PhastCon or PhyloP scores of particular nucleotide positions

        Defined variable:
        parsed_file: list of lists, containing per sequence the PhastCon or PhyloP scores in Strings or NoneTypes
        """

        # PhastCon files are tab delimited, and the scores are located after the second tab
        # The scores are comma separated
        parsed_file = list(map(lambda line: (line.strip('\n').split('\t')[2]).split(','), file))

        # Check on missing values defined with '-', and replace them with NoneType values
        for nt_features in parsed_file[1:]:
            # Only perform action when the feature list of the nucleotide contain missing values
            if '-' in nt_features:
                for index, feature in enumerate(nt_features):
                    if feature == '-':
                        nt_features[index] = None

        # The first list of the parsed file list contains the header line, so this list is excluded
        return parsed_file[1:]

    def gerp_scores(self, file):
        """ Return: list of lists containing per sequence the GerpN, GerpS, GerpRS or Gerp_pval scores in Strings or\
         NoneTypes

        file: open TSV file, containing neutral evolution scores (GerpN) and rejected substitution scores (GerpS), or\
         Gerp element scores (GerpRS) and p-values (Gerp_pval) of particular nucleotide positions

        Defined variables:
        Gerp1: list of lists, containing per sequence the GerpN or GerpRS scores in Strings or NoneTypes
        Gerp2: list of lists, containing per sequence the GerpS or Gerp_pval scores in Strings or NoneTypes
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

    def saving(self, out_array, data_directory, file_name):
        """ Writing the nunmpy array containing the data to H5py format

        out_array: numpy array of shape (number of samples, number of feature, number of nucleotides), contains the\
         final output of this script
        data_directory: string, working directory where the data can be found
        file_name: string, desired file name for the H5py file ending with .py

        Defined variable:
        write_output: HDF5 data file, containing a compressed version of the numpy array from 'out_array'
        """

        # Create a file in the chosen output directory using the latest version of H5py
        h5f = h5py.File(data_directory+file_name, 'w')

        # Write the data to H5py
        # The name of the data set file is equal to the name of the path where the data is coming from
        write_output = h5f.create_dataset(data_directory.split('/')[-2], data=out_array)

        # Check if the data is written properly
        print(write_output)

        # Close the H5py file
        h5f.close()

    def combine(self, data_directory, file_name):
        """ Return numpy array of floats (or Nans) to store the different features scores of a particular sequence.

        data_directory: string, working directory where the data can be found
        file_name: string, desired file name for the H5py file ending with .py

        Defined variables:
        ntA: list of lists, containing boolean numbers to indicate where adenine's are located in a particular sequence
        ntC: list of lists, containing boolean numbers to indicate where cytosine's are located in a particular sequence
        ntT: list of lists, containing boolean numbers to indicate where thymine's are located in a particular sequence
        ntG: list of lists, containing boolean numbers to indicate where guanine's are located in a particular sequence
        phastcon_mam: list of lists, containing Strings (or NoneTypes) which represent mammalian PhastCon scores
        phastcon_pri: list of lists, containing Strings (or NoneTypes) which represent primate PhastCon scores
        phastcon_verp list of lists, containing Strings (or NoneTypes) which represent vertebrate PhastCon scores
        phylop_mam: list of lists, containing Strings (or NoneTypes) which represent mammalian PhyloP scores
        phylop_pri: list of lists, containing Strings (or NoneTypes) which represent primate PhyloP scores
        phylop_verp list of lists, containing Strings (or NoneTypes) which represent vertebrate PhyloP scores
        GerpN: list of lists, containing Strings (or NoneTypes) which represent the neutral evolution scores by GERP++
        GerpS: list of lists, containing Strings (or NoneTypes) which represent rejected substitution scores by GERP++
        GerpRS: list of lists, containing Strings (or NoneTypes) which represent the Gerp element scores
        Gerp_pval: list of lists, containing Strings (or NoneTypes) which represent Gerp element p-values
        input_format: numpy array, containing per sequence booleans for the nucleotides and Strings (or NoneTypes) for\
         the conservation-based scores
        """

        # Get the nucleotides scores
        ntA, ntC, ntT, ntG = self.nt_features(self.genome)
        # Get the PhastCon scores
        phastcon_mam = self.phastcon_phylop_scores(self.phastcon_mam)
        phastcon_pri = self.phastcon_phylop_scores(self.phastcon_pri)
        phastcon_verp = self.phastcon_phylop_scores(self.phastcon_verp)
        # Get the PhyloP scores
        phylop_mam = self.phastcon_phylop_scores(self.phylop_mam)
        phylop_pri = self.phastcon_phylop_scores(self.phylop_pri)
        phylop_verp = self.phastcon_phylop_scores(self.phylop_verp)
        # Get the Gerp scores
        GerpN, GerpS = self.gerp_scores(self.gerp)
        GerpRS, Gerp_pval = self.gerp_scores(self.gerp_elem)

        # Zip the feature scores lists in the way that the scores are sorted per sequence
        zipped = list(zip(ntA, ntC, ntT, ntG,
                          phastcon_mam, phastcon_pri, phastcon_verp,
                          phylop_mam, phylop_pri, phylop_verp,
                          GerpN, GerpS, GerpRS, Gerp_pval))

        # Convert the zipped list into a numpy array of floats
        try:
            input_format = np.array(zipped, dtype='float')
        except:
            pdb.set_trace()

        # Save the numpy array to H5py format
        self.saving(input_format, data_directory, file_name)

        return input_format


if __name__ == "__main__":
    # # Defined data directories for if you do not run from the command line;
    # # all the option-parsers should be switch of then
    # data_directory = "/mnt/scratch/kersj001/data/output/test/test_ben2/"
    # out_directory = "combined_del.h5"

    # Keep track of the running time
    start_time = time.time()

    # Specify the options for running from the command line
    parser = OptionParser()
    # Specify the data directory
    parser.add_option("-p", "--path", dest="path", help="Path to the output of the 'find_sequence.py' script that \
     finds for genomic locations the associated sequences and annotations within a defined window size upstream and \
     downstream of the query location.", default="")
    # Specify the output file name
    parser.add_option("-o", "--output", dest="output", help="Desired file name for the H5py output file ending with \
     .py", default="")

    # Get the command line options
    (options, args) = parser.parse_args()
    data_directory = options.path
    file_name = options.output

    # Run the main function
    inputCNN(data_directory).combine(data_directory, file_name)

    print("----- {} seconds -----".format(round(time.time() - start_time), 2))

    # # Example for running from the command line:
    # python PycharmProjects/thesis/combine_features.py --path /mnt/scratch/kersj001/data/output/1_mil_ben/ \
    #     --output combined_ben.h5
