from pathlib import Path
import argparse
import sys
import csv


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", required=True)
    # parser.add_argument("--exclude", required=True)
    args = parser.parse_args(argv)

    return args


def main(args):
    ex_text = Path(args.exclude).read_text()
    ex_set = set(ex_text.split("\n"))

    reader = csv.reader(sys.stdin, delimiter="|")
    writer = csv.writer(sys.stdout, delimiter="|")

    rows_in = 0
    rows_out = 0
    for row in reader:
        key = row[0]
        if key not in ex_set:
            writer.writerow(row)
            rows_out += 1
        rows_in += 1

    print(f"{rows_in} read, {rows_out} written", file=sys.stderr)


if __name__ == "__main__":
    args = parse_args()
    main(args)
