from collections import Counter
import csv

LABEL_ID = ["/m/05tny_", "/m/07r_k2n", "/m/07qf0zm", "/m/07rc7d9", "/m/0ghcn6", "/t/dd00136", "/m/07srf8z"]
LABEL_NAME = ["Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)", "Bay"]

counter = Counter()
with open('balanced_train_segments.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0].startswith('#'):
            continue
        labels = row[-1].strip().split(',')
        for label in labels:
            counter[label] += 1

for i, label_id in enumerate(LABEL_ID):
    print(f"{LABEL_NAME[i]} ({label_id}): {counter[label_id]} clips")
