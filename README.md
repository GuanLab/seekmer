# Seekmer

Seekmer: a fast RNA-seq analysis tool

The project is hosted on both [GitHub](https://github.com/guanlab/seekmer) and
[GitLab](https://gitlab.com/zhanghjumich/seekmer). We use GitLab for
development purpose and all commits are mirrored to GitHub. Please use GitLab
for reporting issues and posting merge requests.

## Installation

You need Python 3.5+ to use Seekmer. Using [Anaconda](https://www.anaconda.com/distribution/)
is strongly recommended.

To install Seekmer in Anaconda, run:
```bash
conda install -c guanlab seekmer
```

If you prefer to use `pip` to install Seekmer, after you clone the
repository, run:
```bash
pip install <path to cloned Seekmer folder>
```

## Usage

Download ENSEMBL reference genome or reference cDNA files to build index.
You also need to download ENSEMBL GTF annotation.
Then run:
```bash
seekmer index -t <cDNA FASTA file> <GTF file> <output index path>
```
if you are using cDNA sequences, or
```bash
seekmer index <genome DNA FASTA file> <GTF file> <output index path>
```
if you are using genome DNA sequences.

To align bulk-RNA sequences, use
```bash
seekmer infer <generated index file> <output result folder> <FASTQ1> <FASTQ2> ...
```
By default, Seekmer takes pair-ended samples.
For multiple pair-ended samples, the order of FASTQ files should be
`<sample1 end1> <sample1 end2> <sample2 end1> <sample2 end2> ...`.
Add the `-s` option to align single-ended samples.
Add the `-j <number of cores>` option to align in parallel.
Add the `-b <number of bootstraps>` to generate bootstrapped samples.
The output of bulk-RNA sample analysis is compatible to that of Kallisto.

To impute isoforms from single-cell RNA sequences, use
```bash
seekmer impute <generated index file> <output result folder> <FASTQ1> <FASTQ2> ...
```
The folder will contains a correlation file that describe the similarity
between cells, and an imputed isoform abundance.