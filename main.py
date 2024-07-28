import argparse
import torch
from dataset import NS2dDataset
from torch.utils.data import DataLoader
from model import GNOT
from utils import pad_to_max_length
import dgl
from loss import MSELoss, RelL2Loss
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR

def main():
    parser = argparse.ArgumentParser(description="GNOT")

    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--n_attn_layers', type=int, default=4)
    parser.add_argument('--n_attn_hidden_dim', type=int, default=256)
    parser.add_argument('--n_mlp_num_layers', type=int, default=4)
    parser.add_argument('--n_mlp_hidden_dim', type=int, default=256)
    parser.add_argument('--n_input_hidden_dim', type=int, default=256)
    parser.add_argument('--n_expert', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()

    device = torch.device(f'cuda:{str(args.gpu_id)}' if torch.cuda.is_available() else 'cpu')
    train_dataset = NS2dDataset('/home/hui007/GNOT-master/data/ns2d_1100_train.pkl')
    test_dataset = NS2dDataset('/home/hui007/GNOT-master/data/ns2d_1100_test.pkl')
    input_dim = train_dataset[0][0].ndata['x'].shape[1]
    theta_dim = len(train_dataset[0][1])
    input_func_dim = train_dataset[0][2][0].shape[1]
    out_dim = train_dataset[0][0].ndata['y'].shape[1]

    n_input_functions = len(train_dataset[0][2])

    def collate_fn(batch):
        graphs, thetas, input_functions = zip(*batch)
        return graphs, thetas, input_functions

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = GNOT(input_dim, theta_dim, input_func_dim, out_dim, args.n_attn_layers, args.n_attn_hidden_dim, args.n_mlp_num_layers, args.n_mlp_hidden_dim, args.n_input_hidden_dim, args.n_expert, args.n_head, n_input_functions)
    model = model.to(device)

    mse_loss = MSELoss()
    rel_l2_loss = RelL2Loss()

    lr = 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=args.epochs)
    
    best_res = 1e9
    for epoch in range(args.epochs):
        train_loss = []

        for data in train_loader:

            g, theta, input_functions = data
            x = [g[i].ndata['x'] for i in range(len(g))]
            
            max_length = max(tensor.shape[0] for arrays in input_functions for tensor in arrays)
            input_funcs = []

            n = len(input_functions[0])

            for i in range(n):
                tensor_list = []

                for array in input_functions:
                    padded_tensor = pad_to_max_length(array[i], max_length)
                    tensor_list.append(padded_tensor)

                stacked_tensor = torch.stack(tensor_list)
                input_funcs.append(stacked_tensor)

            max_length = max(tensor.shape[0] for tensor in x)
            x = [pad_to_max_length(tensor, max_length) for tensor in x]
            x = torch.stack(x).to(device)
            theta = torch.tensor(theta).float().to(device)
            input_funcs = torch.stack(input_funcs).to(device)

            output = model(x, theta, input_funcs)
            # print(output.shape) 

            g = dgl.batch(list(g))
            g = g.to(device)
            output = torch.cat([output[i, :num] for i, num in enumerate(g.batch_num_nodes())], dim=0)
            output = output.to(device)
            # print(output.shape) 

            target = g.ndata['y']
            target = target.to(device)
            # print(target.shape) 
            
            # loss = mse_loss(g, output, target)
            loss = rel_l2_loss(g, output, target)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch}, Loss: {np.mean(train_loss)}')
        scheduler.step()

        metrics = []
        with torch.no_grad():
            for data in test_loader:
                g, theta, input_functions = data
                x = [g[i].ndata['x'] for i in range(len(g))]
                max_length = max(tensor.shape[0] for arrays in input_functions for tensor in arrays)
                input_funcs = []

                n = len(input_functions[0])

                for i in range(n):
                    tensor_list = []

                    for array in input_functions:
                        padded_tensor = pad_to_max_length(array[i], max_length)
                        tensor_list.append(padded_tensor)

                    stacked_tensor = torch.stack(tensor_list)
                    input_funcs.append(stacked_tensor)

                max_length = max(tensor.shape[0] for tensor in x)
                x = [pad_to_max_length(tensor, max_length) for tensor in x]
                x = torch.stack(x).to(device)
                theta = torch.tensor(theta).float().to(device)
                input_funcs = torch.stack(input_funcs).to(device)

                output = model(x, theta, input_funcs)

                g = dgl.batch(list(g))
                g = g.to(device)
                output = torch.cat([output[i, :num] for i, num in enumerate(g.batch_num_nodes())], dim=0)

                target = g.ndata['y']
                target = target.to(device)

                loss = rel_l2_loss(g, output, target)
                metrics.append(loss.item())

            res = np.mean(metrics)
            print(f'Epoch {epoch}, Test Metric: {res}')
            print('-----------------------------------')
            if res < best_res:
                best_res = res
                torch.save(model.state_dict(), 'best_model.pth')

    print(f'\nBest Test Metric: {best_res}')

if __name__ == "__main__":
    main()