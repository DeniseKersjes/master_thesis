#!/usr/bin/env python
# -*- coding: ASCII -*-

"""
:Author: Christian Gross
:Contact: cgross@tudelft.nl
:Date: 08.12.2017

Adapted by Denise Kersjes
Date: 31.01.2018

-   This script finds for genomic locations the associated sequences and 
    annotations within a defined window size upstream and downstream of the 
    query location.
-   The input has to be a tab delimited file with the query chromsome in the 
    first column and query location in the second. The file containing the 
    genome data and annotations must be tabix indexed and bgzipped.  
-   Output is tab delimited, first column the Chromosome, the second the 
    position and third is the retrieved sequence or annotations. Gerp scores 
    have per position 2 annotations

pip install pyfaidx  
samtools faidx Homo_sapiens_assembly18.fasta 
"""

import os,sys
from optparse import OptionParser
import pysam
from itertools import tee


def input_check(path,label):
	if len(path) == 0:
		print 'No path specified, printing of %s is skipped' %label
		return False
	else:
		return True

def path_check(path):
	if not os.path.exists(path):
		sys.exit('Specified file is no file or does not exist. Parameter: %s' 
                  %(path))

def has_elements(iterator):
	iterator, any_check = tee(iterator)
	try:
		any_check.next()
		return True
	except StopIteration:
		return False

def retrieve_phast_phy(con_tabix,expected_position,chrom,Position,window):
	if has_elements(con_tabix.fetch(chrom,Position-window-1,Position+window)):
		tmp_dict = {}
		outlist = []
		for elements in con_tabix.fetch(chrom,Position-window-1,
                                         Position+window):
			elem = elements.split('\t')
			tmp_dict[int(elem[1])] = elem[-1]
		for position in expected_position:
			try:
				outlist.append(tmp_dict[position])
			except KeyError:
				outlist.append('-')
		return outlist
	else:
		return ['-']*len(expected_position)

def retrieve_Gerp(con_tabix,expected_position,chrom,Position,window):
	if has_elements(con_tabix.fetch(chrom,Position-window-1,Position+window)):
		tmp_dict = {}
		outlist = []
		for elements in con_tabix.fetch(chrom,Position-window-1,
                                         Position+window):
			elem = elements.split('\t')
			tmp_dict[int(elem[1])] = (elem[-2]+','+elem[-1])
		for position in expected_position:
			try:
				outlist.append(tmp_dict[position])
			except KeyError:
				outlist.append('-,-')
		return outlist
	else:
		return ['-,-']*len(expected_position)


parser = OptionParser()
parser.add_option("-i", "--infile", dest="infile", help="Tab delimited genomic\
 location file with chromosome in the first column and position in the second.\
 ",default="")

parser.add_option("-g", "--genome", dest="genome", help="Path to Fasta file of\
 the investigated genome, may be bgzipped",default="")

parser.add_option("-p", "--phastcon", dest="phastcon", help="Path to folder\
 with bgzipped files with phastcon scores for all three multiple sequence\
 alignments",default="")

parser.add_option("-l", "--phylop", dest="phylop", help="Path to folder with\
 bgzipped files with phylop scores for all three multiple sequence alignments"
 ,default="")

parser.add_option("-e", "--gerp", dest="gerp", help="Path to folder with\
 bgzipped files with gerp scores for gerp elements and gerp rejected\
 substitution scores",default="")

parser.add_option("-w", "--window", type=int, dest="window", help="Specifies\
 the window size upstream and downstream of the genomic loctation of interest.\
 default 100",default="100")

parser.add_option("-o", "--out", dest="out", help="Path to folder where the\
 nine output TSV files of this script will be stored", default="")

(options, args) = parser.parse_args()

#get the output directory
path = options.out

#checking if path to file with genomic locations is given and exists 
path_check(options.infile)

#checking if path to genome file is given and if file exists otherwise skip 
#printing upstream-/downstream regions
genome_true = input_check(options.genome,'genomic location')
if genome_true:
	path_check(options.genome)
	genome_out = open(path + 'Genome_out.tsv','w')

#checking if path to phastcon files is given and if files exist
phastcon_true = input_check(options.phastcon,'phastcon')
if phastcon_true:
	#open phastcon_out in current working directory
	priPhastcon_out = open(path + 'priPhastcon_out.tsv','w')
	mamPhastcon_out = open(path + 'mamPhastcon_out.tsv','w')
	verpPhastcon_out = open(path + 'verpPhastcon_out.tsv','w')
	#making sure that the path ends with '/' 
	if (not options.phastcon.endswith('/')):
		options.phastcon = options.phastcon+'/'
	path_check(options.phastcon)
	phastcon_files = [name for name in os.listdir(options.phastcon) if 
                      name.endswith('.gz')]
	if len(phastcon_files) != 3: sys.exit('Phastcon folder does not contain\
 enough files')

#checking if path to phylop files is given and if files exist
phylop_true = input_check(options.phylop,'phylop')
if phylop_true:
	#open phylop_out in current working directory
	priPhylop_out = open(path + 'priPhylop_out.tsv','w')
	mamPhylop_out = open(path + 'mamPhylop_out.tsv','w')
	verpPhylop_out = open(path + 'verpPhylop_out.tsv','w')
	#making sure that the path ends with '/' 
	if (not options.phylop.endswith('/')):
		options.phylop = options.phylop+'/'
	path_check(options.phylop)
	phylop_files = [name for name in os.listdir(options.phylop) if 
                    name.endswith('.gz')]
	if len(phylop_files) != 3: sys.exit('Phylop folder does not contain\
 enough files')

#checking if path to gerp files is given and if files exist
gerp_true = input_check(options.gerp,'gerp')
if gerp_true:
	#open gerp_out in current working directory
	gerp_out = open(path + 'Gerp_out.tsv','w')
	gerpElem_out = open(path + 'GerpElem_out.tsv','w')
	#making sure that the path ends with '/' 
	if (not options.gerp.endswith('/')):
		options.gerp = options.gerp+'/'
	path_check(options.gerp)
	gerp_files = [name for name in os.listdir(options.gerp) if 
                  name.endswith('.gz')]
	if len(gerp_files) != 2: sys.exit('Gerp folder does not contain enough\
 files')

#if genome given, open .gz tabix fasta file
if genome_true:
	ref_fasta = pysam.Fastafile(options.genome)

#if phastcon given, open .gz tabix files
if phastcon_true:
	for file in phastcon_files:
		if 'pri' in file: 
			priPhastCon_tabix = pysam.Tabixfile(options.phastcon+file,'r')
		elif 'pla' in file:
			mamPhastCon_tabix = pysam.Tabixfile(options.phastcon+file,'r')
		elif 'ver' in file:
			verPhastCon_tabix = pysam.Tabixfile(options.phastcon+file,'r')

#if phylop given, open .gz tabix files
if phylop_true:
	for file in phylop_files:
		if 'pri' in file: 
			priPhylop_tabix = pysam.Tabixfile(options.phylop+file,'r')
		elif 'pla' in file:
			mamPhylop_tabix = pysam.Tabixfile(options.phylop+file,'r')
		elif 'ver' in file:
			verPhylop_tabix = pysam.Tabixfile(options.phylop+file,'r')

#if gerp given, open .gz tabix files
if gerp_true:
	for file in gerp_files:
		if 'scores' in file: 
			Gerp_tabix = pysam.Tabixfile(options.gerp+file,'r')
		elif 'elem' in file:
			GerpElem_tabix = pysam.Tabixfile(options.gerp+file,'r')

#writing headers of files
genome_out.write('#Chrom\tPos\tSeq_%s\n'%(options.window))
priPhastcon_out.write('#Chrom\tPos\tpriPhastCon_%s\n'%(options.window))
mamPhastcon_out.write('#Chrom\tPos\tmamPhastCon_%s\n'%(options.window))
verpPhastcon_out.write('#Chrom\tPos\tverPhastCon_%s\n'%(options.window))
priPhylop_out.write('#Chrom\tPos\tpriPhylop_%s\n'%(options.window))
mamPhylop_out.write('#Chrom\tPos\tmamPhylop_%s\n'%(options.window,))
verpPhylop_out.write('#Chrom\tPos\tverPhylop_%s\n'%(options.window))
gerp_out.write('#Chrom\tPos\t[GerpN,GerpS]_%s\n'%(options.window))
gerpElem_out.write('#Chrom\tPos\t[GerpRS,Gerppval]_%s\n'%(options.window))


infile = open(options.infile,'r')

for lines in infile:
	if lines.startswith('#'):
		continue
	else:
		line = lines.strip().split('\t')
		chr = line[0]
		Pos = int(line[1])
		expected_pos = range(Pos-options.window,Pos+options.window+1)
		
		if genome_true:
			sequence = ref_fasta.fetch(chr,Pos-options.window-1,
                                        Pos+options.window)
			genome_out.write(chr+'\t'+line[1]+'\t'+sequence+'\n')
		if phastcon_true:
			priPhastcon_out.write(chr+'\t'+line[1]+'\t'+','.join(
                   retrieve_phast_phy(priPhastCon_tabix,expected_pos,chr,Pos,
                                      options.window))+'\n')
			mamPhastcon_out.write(chr+'\t'+line[1]+'\t'+','.join(
                   retrieve_phast_phy(mamPhastCon_tabix,expected_pos,chr,Pos,
                                      options.window))+'\n')
			verpPhastcon_out.write(chr+'\t'+line[1]+'\t'+','.join(
                   retrieve_phast_phy(verPhastCon_tabix,expected_pos,chr,Pos,
                                      options.window))+'\n')
		if phylop_true:
			priPhylop_out.write(chr+'\t'+line[1]+'\t'+','.join(
                   retrieve_phast_phy(priPhylop_tabix,expected_pos,chr,Pos,
                                      options.window))+'\n')
			mamPhylop_out.write(chr+'\t'+line[1]+'\t'+','.join(
                   retrieve_phast_phy(mamPhylop_tabix,expected_pos,chr,Pos,
                                      options.window))+'\n')
			verpPhylop_out.write(chr+'\t'+line[1]+'\t'+','.join(
                   retrieve_phast_phy(verPhylop_tabix,expected_pos,chr,Pos,
                                      options.window))+'\n')
		if gerp_true:
			gerp_out.write(chr+'\t'+line[1]+'\t'+','.join(retrieve_Gerp(
                   Gerp_tabix,expected_pos,chr,Pos,options.window))+'\n')
			gerpElem_out.write(chr+'\t'+line[1]+'\t'+','.join(retrieve_Gerp(
                   GerpElem_tabix,expected_pos,chr,Pos,options.window))+'\n')
