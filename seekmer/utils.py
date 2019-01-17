#!/usr/bin/env python3

import argparse
import os
import pathlib
import random


from . import index_builder


_NEWLINE = os.linesep.encode()

_BASES = [b'A', b'T', b'C', b'G']


def main():
    parser = argparse.ArgumentParser(description='simulate fragment sequences',
                                     add_help=True,
                                     formatter_class=_SubcommandHelpFormatter)
    parser.add_argument('fasta_path', type=pathlib.Path, metavar='fasta',
                        help='specify a cDNA sequence file')
    parser.add_argument('fastq1_path', type=pathlib.Path, metavar='fastq1',
                        help='specify a output FASTQ file')
    parser.add_argument('fastq2_path', type=pathlib.Path, metavar='fastq2',
                        help='specify a paired output FASTQ file')
    parser.add_argument('count', type=int, metavar='count',
                        help='specify the number of reads')
    parser.add_argument('length', type=int, metavar='length',
                        help='specify the length of reads')
    opts = vars(parser.parse_args())
    generate_sequences(**opts)


def generate_sequences(fasta_path, fastq1_path, fastq2_path, count, length,
                       *__, **___):
    transcript_ids, sequences = index_builder.read_transcripts(fasta_path)
    tail = _NEWLINE + b'+' + _NEWLINE + b'#' * length + _NEWLINE
    with fastq1_path.open('wb') as fastq1, fastq2_path.open('wb') as fastq2:
        for i in range(count):
            id_, read1 = random_fragment(length, transcript_ids, sequences)
            read2 = random_sequence(length)
            if random.random() < 0.5:
                read1, read2 = read2, read1
                insert = 2
            else:
                insert = 1
            head = ('@read{} {} in read{}'.format(i, id_, insert).encode()
                    + _NEWLINE)
            fastq1.write(head + read1 + tail)
            fastq2.write(head + read2 + tail)


class _SubcommandHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """A formatting class to hide the subcommand title."""

    def _format_action(self, action):
        parts = super()._format_action(action)
        if action.nargs == argparse.PARSER:
            parts = "\n".join(parts.split("\n")[1:])
        return parts


def random_sequence(length):
    return b''.join(random.choice(_BASES) for __ in range(length))


def random_fragment(length, transcript_ids, sequences):
    insert_start = random.randrange(0, length)
    insert_end = random.randrange(0, length)
    if insert_start > insert_end:
        insert_start, insert_end = insert_end, insert_start
    insert_length = insert_end - insert_start
    index = random.randrange(0, len(transcript_ids))
    id_ = transcript_ids[index]
    offset = random.randrange(0, len(sequences[index]))
    fragment = b''.join(
        [random.choice(_BASES) for __ in range(insert_start)]
        + [sequences[index][offset:(offset + insert_length)]]
        + [random.choice(_BASES) for __ in range(length - insert_end)]
    )
    return '{} {}:{}'.format(id_.decode(), insert_start, insert_end), fragment


if __name__ == '__main__':
    main()
