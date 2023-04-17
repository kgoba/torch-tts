from collections import defaultdict
import sys
import unicodedata


def separated_reader(stream, separator, field_idx=None):
    for line in open(args.dataset):
        line = line.strip()
        fields = line.split(separator)
        if field_idx == None:
            yield fields
        else:
            yield [x for (i, x) in enumerate(fields) if i in field_idx]


def main(args):
    cat_dict = defaultdict(set)
    count_dict = defaultdict(int)
    for utt in separated_reader(open(args.dataset), "|", [1]):
        utt = utt[0]
        for c in utt:
            count_dict[c] += 1
            cat = unicodedata.category(c)
            cat_dict[cat].add(c)

    for cat in cat_dict:
        s = sorted(cat_dict[cat])
        print(cat, len(s), "".join(s))

    rare_c = []
    for c in count_dict:
        if count_dict[c] < 5:
            rare_c.append(c)
    print(f"Rare chars: {''.join(sorted(rare_c))}")
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset path")
    args = parser.parse_args()
    rc = main(args)
    sys.exit(rc)
