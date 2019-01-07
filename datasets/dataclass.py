import pickle
import numpy as np

class siamese:

    # Create model
    def __init__(self):
        self.data = open("./datasets/data.pickle", 'r+b')
        self.percentage_matching = 50
        self.set = "custom"
    def __del__(self):
        self.data.close()
    def set_ratio(self, ratio):
        self.match_ratio = ratio
    def set_type(self, type):
        assert type in self.data_dict
        self.set = type
    def populate_data(self, color, custom, gray):
        self.data_dict = {"color":color, "custom":custom, "gray":gray}
    def save_set(self):
        pickle.dump(self.data_dict, self.data)
    def load_set(self):
        self.data_dict = pickle.load(self.data)
    def get_batch(self, batch_size, dataset='custom'):
        n = len(self.data_dict[dataset])
        batch_a = []
        batch_b = []
        batch_bool = []
        for i in range(batch_size):
            if np.random.random_integers(0, 99, 1) < self.percentage_matching:
                temp = self.get_matched_pair(dataset)
            else:
                temp = self.get_unmatched_pair(dataset)
            batch_a.append(temp[0])
            batch_b.append(temp[1])
            batch_bool.append(temp[2])
        return [batch_a, batch_b, batch_bool]
    def get_matched_pair(self, dataset='custom'):
        index = np.random.randint(0,len(self.data_dict[dataset]))
        [image_a, image_b] = np.random.choice(len(self.data_dict[dataset][index]), 2, replace=False)
        return [self.data_dict[dataset][index][image_a], self.data_dict[dataset][index][image_b], True]
    def get_unmatched_pair(self, dataset='custom'):
        [index_a, index_b] = np.random.choice(len(self.data_dict[dataset]), 2, replace=False)
        [image_a, image_b] = np.random.choice(len(self.data_dict[dataset][index_a]), 2, replace=False)
        return [self.data_dict[dataset][index_a][image_a], self.data_dict[dataset][index_b][image_b], False]

    """
    def get_batch(self, batch_size):
        n = self.data_dict[self.type].shape[0]
        batch = np.floor(np.random.rand(batch_size) * n).astype(int)
        batch_x = self.data_dict[self.type][batch, :]
        batch_y = Ydata[batch]
        return [batch_x, batch_y]

    def get_paired_batches(self, xdata, ydata):
        percentage_matching = 50
        [batch1_x, batch1_y] = get_batch(xdata, ydata, 128)
        [raw2_x, raw2_y] = get_batch(xdata, ydata, 256)
        max_y = max(raw2_y)

        separated = [[] for k in range(max_y + 1)]
        for (x, y) in zip(raw2_x, raw2_y):
            separated[y].append(x)
        batch2_x = []
        batch2_y = []
        for (x, y) in zip(batch1_x, batch1_y):
            if len(batch2_y) != len(batch2_x):
                print("test")
            if np.random.random_integers(0, 99, 1) < percentage_matching:
                # get one from the corresponding separated
                batch2_x.append(separated[y][0])
                batch2_y.append(y)
                if len(separated[y]) > 1:
                    del separated[y][0]
            else:
                ind = y
                while (ind == y):
                    ind = np.random.random_integers(0, max_y)
                # get one from another section of separated
                batch2_x.append(separated[ind][0])
                batch2_y.append(ind)
                if len(separated[ind]) > 1:
                    del separated[ind][0]
        batch2_x = np.asarray(batch2_x)
        batch2_y = np.asarray(batch2_y)
        return [batch1_x, batch1_y, batch2_x, batch2_y]
    """
    
