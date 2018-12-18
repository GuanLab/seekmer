#!/usr/bin/env python3

import argparse
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import logbook

from . import index_builder
from . import mapper


def main():
    """The main entrypoint of Seekmer"""
    parser = argparse.ArgumentParser(description='A fast RNA-seq tool',
                                     add_help=True,
                                     formatter_class=_SubcommandHelpFormatter)
    parser.add_argument('-v', '--version', action='version',
                        version='Seekmer 2019.0.0')
    parser.add_argument('--debug', action='store_true',
                        help='enable debugging messages')
    subparsers = parser.add_subparsers(title='subcommand', dest='subcommand')
    index_builder.add_subcommand_parser(subparsers)
    mapper.add_subcommand_parser(subparsers)
    opts = vars(parser.parse_args())
    log_handler = initialise_logging(opts)
    with log_handler.applicationbound():
        if opts['subcommand'] == 'index':
            index_builder.run(**opts)
        elif opts['subcommand'] == 'infer':
            mapper.run(**opts)
        else:
            parser.print_help()


class _SubcommandHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """A formatting class to hide the subcommand title."""

    def _format_action(self, action):
        parts = super()._format_action(action)
        if action.nargs == argparse.PARSER:
            parts = "\n".join(parts.split("\n")[1:])
        return parts


def initialise_logging(opts):
    """Initialize a logging handler.

    Enable debugging output if `opts['debug']` is `True`.

    Parameters
    ----------
    opts : dict
        Command line options

    Returns
    -------
    logbook.Handler
        A logging handler
    """
    log_handler = logbook.StderrHandler(
        level=('DEBUG' if opts['debug'] else 'INFO'),
    )
    log_handler.format_string = (
        '{record.level_name:<5} {record.time:%Y-%m-%d %H:%M:%S} '
        '{record.channel}: {record.message}'
    )
    return log_handler


if __name__ == '__main__':
    sys.exit(main())
