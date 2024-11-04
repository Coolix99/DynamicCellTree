from collections import defaultdict

class QTUnionFind:
    def __init__(self, size):
        self.parent = [-1] * size
        self.rank = [0] * size
        self.size = 0

    def make_set(self, q):
        self.parent[self.size] = -1
        self.rank[self.size] = 0
        self.size += 1

    def find_canonical(self, q):
        r = q
        while self.parent[r] >= 0:
            r = self.parent[r]
        while self.parent[q] >= 0:
            tmp = q
            q = self.parent[q]
            self.parent[tmp] = r
        return r

    def union(self, cx, cy):
        if self.rank[cx] > self.rank[cy]:
            cx, cy = cy, cx
        if self.rank[cx] == self.rank[cy]:
            self.rank[cy] += 1
        self.parent[cx] = cy
        return cy

class QEBTUnionFind:
    def __init__(self, size):
        self.QBT_parent = [-1] * (2 * size - 1)
        self.QBT_children = defaultdict(list)
        self.QBT_size = size
        self.QT = QTUnionFind(2 * size - 1)
        self.Root = list(range(size))
        self.split_times = {}  # To store separation times for each new parent

    def make_set(self, q):
        self.Root[q] = q
        self.QBT_parent[q] = -1
        self.QT.make_set(q)

    def find_canonical(self, q):
        return self.QT.find_canonical(q)

    def union(self, cx, cy, sep_time):
        tu = self.Root[cx]
        tv = self.Root[cy]
        new_parent = self.QBT_size
        self.QBT_size += 1
        self.QBT_parent[tu] = new_parent
        self.QBT_parent[tv] = new_parent
        self.QBT_children[new_parent].extend([tu, tv])

        combined_root = self.QT.union(cx, cy)
        self.Root[combined_root] = new_parent
        self.QBT_parent[new_parent] = -1

        # Store separation time for the new parent node
        self.split_times[new_parent] = sep_time

        return new_parent

def build_trees_from_splits(splits):
    unique_labels = set()
    for (label1, label2), _ in splits.items():
        unique_labels.add(label1)
        unique_labels.add(label2)

    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    biggest_label = max(unique_labels)
    size = len(unique_labels)
    qebt = QEBTUnionFind(size=size)

    for (label1, label2), sep_time in sorted(splits.items(), key=lambda x: x[1], reverse=True):
        idx1 = label_to_index[label1]
        idx2 = label_to_index[label2]
        cx = qebt.find_canonical(idx1)
        cy = qebt.find_canonical(idx2)
        if cx != cy:
            qebt.union(cx, cy, sep_time)

    def construct_forest():
        connections = defaultdict(dict)
        roots = []

        for idx, parent in enumerate(qebt.QBT_parent):
            if parent != -1:
                parent_label = index_to_label.get(parent, parent+biggest_label-size+1)
                child_label = index_to_label.get(idx, idx+biggest_label-size+1)
                connections[parent_label].setdefault("children", []).append(child_label)
                connections[parent_label]["split_time"] = qebt.split_times.get(parent, None)
            else:
                roots.append(index_to_label.get(idx, idx+biggest_label-size+1))

        roots = [root for root in roots if root in connections]

        return dict(connections), roots
    print(qebt.QBT_parent)
    return construct_forest()