import os
import sys
from argparse import ArgumentParser
from fileutils import get_rts_from_xmlfile
from graphutils import *


def get_args():
    """ Command line arguments """
    parser = ArgumentParser(description="Generate CPPs")
    parser.add_argument("file", help="XML file with RTS", type=str)
    parser.add_argument("--id", help="STR in file to process", type=int, default=0)
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.isfile(args.file):
        print("Can't find {0} file.".format(args.xmlfile))
        sys.exit(1)

    path, ext = os.path.splitext(args.file)

    if ext == ".xml":
        rts = get_rts_from_xmlfile(args.id, args.file)
        graph = create_graph_from_rts(rts)
        save_graph_img(graph, path)
    elif ext == ".dot":
        from fileutils import get_rts_from_pot_file
        _, graph = get_rts_from_pot_file(args.file)
        save_graph_img(graph, path)
    else:
        print("Unknown extension... *.dot or *.xml please!")


if __name__ == '__main__':
    main()