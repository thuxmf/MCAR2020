import torch.utils.data as data
import os
import os.path
import numpy as np


def make_dataset(dir):
    datas = []
    for fname in os.listdir(dir):
        if 'npy' in fname:
            path = os.path.join(dir, fname)
            item = (path, 0)
            datas.append(item)

    return datas


class DatasetFolder(data.Dataset):
    def __init__(self, root):
        # classes, class_to_idx = find_classes(root)
        samples = make_dataset(root)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

        self.root = root
        self.samples = samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = np.load(path, allow_pickle=True).item()

        return sample, target

    def __len__(self):
        return len(self.samples)


class DataFolder(DatasetFolder):
    def __init__(self, root):
        super(DataFolder, self).__init__(root)
        self.datas = self.samples


