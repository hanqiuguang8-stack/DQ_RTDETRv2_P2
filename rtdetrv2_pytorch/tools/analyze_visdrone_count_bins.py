import argparse
import json
from collections import Counter

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', required=True, help='COCO format VisDrone train annotation json')
    args = parser.parse_args()

    with open(args.ann, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    count_dict = {img['id']: 0 for img in coco['images']}
    for ann in coco['annotations']:
        if ann.get('iscrowd', 0) == 1:
            continue
        img_id = ann['image_id']
        if img_id in count_dict:
            count_dict[img_id] += 1

    counts = np.asarray(list(count_dict.values()), dtype=np.int64)
    print('num_images:', len(counts))
    print('min:', int(counts.min()))
    print('max:', int(counts.max()))
    print('mean:', float(counts.mean()))
    print('std:', float(counts.std()))
    for p in (25, 33, 50, 66, 75, 90, 95):
        print(f'p{p}:', float(np.percentile(counts, p)))

    bins_3 = [int(np.percentile(counts, 33)), int(np.percentile(counts, 66))]
    bins_4 = [
        int(np.percentile(counts, 25)),
        int(np.percentile(counts, 50)),
        int(np.percentile(counts, 75)),
    ]
    print('suggested count_bins_3:', bins_3)
    print('suggested count_bins_4:', bins_4)

    hist = Counter(counts.tolist())
    print('top count frequencies:')
    for k, v in hist.most_common(20):
        print(k, v)


if __name__ == '__main__':
    main()
