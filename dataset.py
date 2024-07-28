from dgl.data import DGLDataset
import pickle
import dgl
import torch

class NS2dDataset(DGLDataset):
    # [X, Y, theta, (f1, f2, ...)]
    def __init__(self, path):
        with open(path, 'rb') as file:
            self.data = pickle.load(file)
        self.graphs = []
        self.thetas = []
        self.input_functions = []
        
        super().__init__(name='ns2d_dataset')

    def process(self):
        for i in range(len(self.data)):
            x = self.data[i][0]
            y = self.data[i][1]
            g = dgl.DGLGraph()
            g.add_nodes(x.shape[0])
            g.ndata['x'] = torch.from_numpy(x).float()
            g.ndata['y'] = torch.from_numpy(y).float()
            # g.ndata['x'] = torch.from_numpy(x)
            # g.ndata['y'] = torch.from_numpy(y)
            self.graphs.append(g)

            # self.thetas.append(torch.from_numpy(self.data[i][2]).float())
            self.thetas.append(self.data[i][2])
            # self.input_functions.append(self.data[i][3])
            input_function = self.data[i][3]
            new_input_function = []
            if input_function:
                for j in range(len(input_function)):
                    new_input_function.append(torch.from_numpy(input_function[j]).float())
                    # new_input_function.append(torch.from_numpy(input_function[j]))
                self.input_functions.append(new_input_function)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.graphs[idx], self.thetas[idx], self.input_functions[idx]

# test:
# train_dataset = NS2dDataset('/home/hui007/GNOT-master/data/ns2d_1100_train.pkl')
# print(train_dataset[0])
# print(len(train_dataset[0][1]))