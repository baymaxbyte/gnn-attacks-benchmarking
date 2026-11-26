import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix, random
from scipy.stats import rv_continuous
import argparse
import subprocess

# Ensure the code runs on GPU
# Change relative to the provided script: ONLY the C-update is label-free.
# Classification is still trained/evaluated using node labels.
r_global = 0
dataset_global = ""
device_global = ""
detailed = 0

def get_gpu_with_max_memory():
    try:
        # Query nvidia-smi to get GPU memory information
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        
        # Parse the output and get available memory for each GPU
        memory_free = [int(x) for x in result.strip().split('\n')]
        
        # Get the GPU index with the maximum available memory
        gpu_index = memory_free.index(max(memory_free))
        
        return gpu_index, max(memory_free)
    except Exception as e:
        print(f"Failed to get GPU memory information: {e}")
        return None, None

def set_best_gpu():
    gpu_index, max_memory = get_gpu_with_max_memory()
    if gpu_index is not None:
        torch.cuda.set_device(gpu_index)
        print(f"Using GPU {gpu_index} with {max_memory} MiB free memory.")
        return torch.device(f'cuda:{gpu_index}')
    else:
        print("No suitable GPU found. Using default GPU.")
        return torch.device('cuda')


def check_cuda_availability():
    if torch.cuda.is_available():
        print("CUDA is available on this device.")
        print(f"Number of available CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available on this device.")

def terminal_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--device', type=str, required=True, help='Device')
    parser.add_argument('--r', type=float, required=True, help='Coarsening ratio')
    parser.add_argument('--max-trials', type=int, default=None, help='Limit number of grid settings (for testing)')
    parser.add_argument('--epochs', type=int, default=500, help='GCN training epochs (default 500)')
    parser.add_argument('--iterations', type=int, default=10, help='C-update iterations (default 10)')
    parser.add_argument('--repeats', type=int, default=2, help='GCN training repetitions (default 2)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    args = parser.parse_args()
    
    global r_global, dataset_global, device_global, args_global
    args_global = args
    if not 0 < args.r < 1:
        parser.error("--r must satisfy 0 < r < 1")
    if args.epochs < 3 or args.iterations < 1 or args.repeats < 1:
        parser.error("--epochs >= 3, --iterations >= 1, and --repeats >= 1 required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset_global = args.dataset
    device_global = args.device
    r_global = args.r

    update_paths()

def update_paths():
    global readings_path, path
    path = os.getcwd()
    readings_path = os.path.join(path, dataset_global, str(r_global))
    os.makedirs(readings_path, exist_ok=True)
    
    print(f"Readings path: {readings_path}")
    ensure_data_frame_exists()

def ensure_data_frame_exists():
    global filepath, filename
    filename = f"parameter({r_global},{dataset_global},{device_global},ENTROPY)_NoY_In_C_Update.csv"
    filepath = os.path.join(readings_path, filename)
    try:
        pd.read_csv(filepath)
    except FileNotFoundError:
        print("Dataframe is being made.")
        df = pd.DataFrame(columns=['alpha_param','beta_param','gamma_param','r','k','acc','std'])
        df.to_csv(filepath, index=False)

def save_readings(alpha_param, beta_param, gamma_param, r, k, acc, std):
    df = pd.read_csv(filepath)
    df.loc[len(df)] = [alpha_param, beta_param, gamma_param, r, k, acc, std]
    df.to_csv(filepath, index=False)

def get_device(device_global):
    if device_global.upper() == 'GPU':
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is not available, but GPU was requested.")
        return set_best_gpu()
    elif device_global.upper() == 'CPU':
        return torch.device('cpu')
    elif device_global.lower().startswith('cuda:'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        return torch.device(device_global)
    else:
        raise ValueError('--device must be CPU, GPU, or cuda:0')

if __name__ == "__main__":
    check_cuda_availability()
    terminal_command()
    device = get_device(device_global)
    print(f"Using device: {device}")

data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)
dataset_name = {'cora': 'Cora', 'citeseer': 'CiteSeer', 'pubmed': 'PubMed'}.get(dataset_global.lower())
if dataset_name is None:
    raise ValueError('--dataset must be Cora, CiteSeer, or PubMed')
dataset = Planetoid(root='data/planetoid', name=dataset_name)

edge_list = dataset[0].edge_index
NO_OF_EDGES = edge_list.shape[1]
labels = dataset[0].y.to(device)

adj = to_dense_adj(dataset[0].edge_index).to(device)
adj = adj[0]

X = dataset[0].x.to_dense().to(device)
N = X.shape[0]
NO_OF_CLASSES = len(set(labels.cpu().numpy()))

sparsity_original = 2 * NO_OF_EDGES / (N * (N - 1))

nn = int(1 * N)
X = X[:nn, :]
adj = adj[:nn, :nn]
labels = labels[:nn]

def get_laplacian(adj):
    b = torch.ones(adj.shape[0], device=device)
    return torch.diag(adj @ b) - adj

theta = get_laplacian(adj).float()

p = X.shape[0]
k = int(p * r_global)
k_global = int(p * r_global)
if k < 10:
    raise ValueError('Coarsening ratio too small: need k >= 10 for the classifier split.')
n = X.shape[1]
beta_param = 50
alpha_param = 100
gamma_param = 100
lr = 1e-5
thresh = 1e-10

class CustomDistribution(rv_continuous):
    def _rvs(self, size=None, random_state=None):
        return random_state.standard_normal(size)
temp = CustomDistribution(seed=1)
temp2 = temp()
# X_tilde = torch.tensor(random(k, n, density=0.25, random_state=1, data_rvs=temp2.rvs).toarray(), dtype=torch.float32, device=device)
# C = torch.tensor(random(p, k, density=0.25, random_state=1, data_rvs=temp2.rvs).toarray(), dtype=torch.float32, device=device)

total_indices1 = torch.arange(N, device=device)
shuffled_indices1 = torch.randperm(total_indices1.numel(), device=device)

# Create a training mask
train_size = int(0.1 * total_indices1.numel())
train_indices = shuffled_indices1[:train_size]
train_mask = torch.zeros(N, dtype=torch.bool, device=device)
train_mask[train_indices] = True

print(f"train size: {train_size}")

# Create a test mask
# test_indices = shuffled_indices1[train_size:]
# test_mask = torch.zeros(N, dtype=torch.bool, device=device)
# test_mask[test_indices] = True

# print(f"test mask: {test_mask.shape}")

def convertScipyToTensor(coo):
    coo = coo.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices).to(device)
    v = torch.FloatTensor(values).to(device)
    shape = coo.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape)).to(device)
                          
def experiment(alpha_param,beta_param,gamma_param,C,theta,X, exp_iter):
      p = X.shape[0]
      k = int(p* r_global)
      n = X.shape[1]
      ones = csr_matrix(np.ones((k,k)))
      ones = convertScipyToTensor(ones).to(device)
      ones = ones.to_dense()
      
      
        
      try:
        C = convertScipyToTensor(C)
        C = C.to_dense()
      except:
        C=C
      try:
        theta = convertScipyToTensor(theta)
      except:
        theta = theta
      try:
        X = convertScipyToTensor(X)
        X = X.to_dense()
      except:
        X = X
      C, theta, X, ones = (item.to(device) for item in (C, theta, X, ones))
      def update_U(C,theta):
          CT= torch.transpose(C,0,1)
          lamb,U=torch.linalg.eigh((CT @ theta @ C).double())
          return U.flip(dims=(1,))   
      def bracket_term2fun(C,CT,theta):
          U  = update_U(C,theta).double() 
          UT= torch.transpose(U,0,1)
          Lw = (CT @theta @C).double()
          k_ = min(7, Lw.size(0) - 1)  # preserve original spectral constraint
          lb= 1e-5
          ub = 1e+4
          beta = 0.5 
          lambda_ =  laplacian_lambda_update(lb, ub, beta, U, Lw, k_)   
          lambda_matrix =  torch.diag(lambda_,0).to(device)
          return U@lambda_matrix@UT

      def update_C(C):
#           thetaC = theta@C
          CT = torch.transpose(C,0,1)
          t1 = alpha_param*(C@ones)
          bracket_term1 = (CT@theta@C)
          bracket_term2 = bracket_term2fun(C,CT,theta) 
          bracket_term = bracket_term1 - bracket_term2   # bracket term (CT*theta*C - U*lambda*UT)
          t2 = beta_param*(theta@C@bracket_term.float())
          # NO LABELS in C optimization. The original Y/P/t6 term is removed.
          # GCN classification below still uses labels and computes accuracy.
          grad_fc= t1+t2
          C_new=C-gamma_param*grad_fc
          C_new[C_new<thresh] = thresh
          C_new = C_new / C_new.sum(dim=1, keepdim=True).clamp_min(thresh)
          return C_new        
            

        
        
        


      #We set c1 = 10−5 and c2 = 10^4 We observed that the experimental performances of the algorithms 
       #are not sensitive to different values of c1 and c2 as long as they are reasonably small and large,respectively
      # K is the number of smallest eigenvalues of the Laplacian matrix that are being ignored while updating the eigenvalues.
      def laplacian_lambda_update(lb, ub, beta, U, Lw, k):
        q = max(0, Lw.size(1) - k)
        UT= torch.transpose(U,0,1)
        UT = UT.type(torch.float64)
        d = torch.diag(UT @ Lw @ U)
        # unconstrained solution as initial point
        lambda_ = 0.5 * (d + torch.sqrt(d.pow(2) + 4 / beta))
        lambda_,indices = torch.sort(lambda_, dim=- 1, descending=True)
        eps = 1
        condition = bool(((lambda_[q] - ub) <= eps) & ((lambda_[0] - lb) >= -eps))
#                                   (lambda_[1:(q)] - lambda_[0:(q-1)]) >= -eps])
        if condition:
            return lambda_
        else:
            greater_ub = lambda_ > ub
            lesser_lb = lambda_ < lb
            lambda_[greater_ub] = ub
            lambda_[lesser_lb] = lb
            condition = bool(((lambda_[q] - ub) <= eps) & ((lambda_[0] - lb) >= -eps))
#                                   (lambda_[1:q] - lambda_[0:(q-1)]) >= -eps])
            if condition:
                return lambda_
            else:
                print(lambda_)
                raise ValueError("eigenvalues are not in increasing order, consider increasing the value of beta")
            

      for i in tqdm(range(exp_iter)): #update C only 21
         C = update_C(C)
            
      return C


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(X.shape[1], 64)
        self.conv2 = GCNConv(64, NO_OF_CLASSES)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)

def get_accuracy(C_0, L):
    global labels, NO_OF_CLASSES, k
    all_acc = []
    for _ in range(args_global.repeats):
        C_0_new = torch.zeros_like(C_0)
        C_0_new.scatter_(1, C_0.argmax(dim=1, keepdim=True), 1)

        Lc = C_0_new.T @ L @ C_0_new
        Wc = (-1 * Lc) * (1 - torch.eye(Lc.shape[0], device=device))
        Wc[Wc < 0.1] = 0

        coarse_edge_index, coarse_edge_weight = dense_to_sparse(Wc)

        def one_hot(x, class_count):
            return torch.eye(class_count, device=device)[x, :]

        Y = labels.to(device)
        Y = one_hot(Y, NO_OF_CLASSES)
        Y[train_mask, :] = 0

        P = torch.pinverse(C_0_new)
        labels_coarse = torch.argmax(torch.sparse.mm(P, Y), 1)

        Xt = P @ X.to(device)
        coarsen_features = Xt
        coarsen_train_labels = labels_coarse

        total_indices = torch.arange(k_global, device=device)
        shuffled_indices = torch.randperm(total_indices.numel(), device=device)
        train_size = int(0.8 * total_indices.numel())
        train_indices = shuffled_indices[:train_size]
        val_size = int(0.2 * total_indices.numel())
        val_indices = shuffled_indices[train_size:train_size + val_size]

        data = dataset[0].to(device)

        coarsen_train_mask = train_indices
        coarsen_val_labels = labels_coarse
        coarsen_val_mask = val_indices

        model = Net().to(device)
        lr = 0.01
        decay = 0.0001
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

        model.reset_parameters()
        pathm = path + "/"
        best_val_loss = float('inf')
        no_of_epochs = args_global.epochs
        no_of_early_stopping = 10
        val_loss_history = []

        for epoch in range(no_of_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(coarsen_features, coarse_edge_index, coarse_edge_weight)
            loss = F.nll_loss(out[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            pred = model(coarsen_features, coarse_edge_index, coarse_edge_weight)
            val_loss = F.nll_loss(pred[coarsen_val_mask], coarsen_val_labels[coarsen_val_mask]).item()

            if val_loss < best_val_loss and epoch > no_of_epochs // 2:
                best_val_loss = val_loss
                torch.save(model.state_dict(), pathm + 'checkpoint-best-acc-noY-C.pkl')

            val_loss_history.append(val_loss)
            if no_of_early_stopping > 0 and epoch > no_of_epochs // 2:
                tmp = torch.tensor(val_loss_history[-(no_of_early_stopping + 1):-1], device=device)
                if val_loss > tmp.mean().item():
                    break

        model.load_state_dict(torch.load(pathm + 'checkpoint-best-acc-noY-C.pkl', map_location=device, weights_only=True))
        model.eval()
        pred = model(data.x.to(device), data.edge_index.to(device)).max(1)[1]
        test_acc = int(pred.eq(data.y.to(device)).sum().item()) / data.y.size(0)
        print("Test_acc", f"{test_acc:.4f}")
        all_acc.append(test_acc)

    print('ave_acc: {:.8f}'.format(torch.mean(torch.tensor(all_acc))), '+/- {:.8f}'.format(torch.std(torch.tensor(all_acc), correction=0)))
    return torch.mean(torch.tensor(all_acc)), torch.std(torch.tensor(all_acc), correction=0)

def fitness_function(alpha_param, beta_param, gamma_param):
    print(alpha_param, beta_param, gamma_param)
    acc, std = torch.tensor(0), torch.tensor(0)
    start_time = time.time()  # Start time for the test case
    try:
        # X_tilde = torch.tensor(random(k, n, density=0.15, random_state=1, data_rvs=temp2.rvs).toarray(), device=device, dtype=torch.float32)
        C = torch.tensor(random(p, k, density=0.15, random_state=1, data_rvs=temp2.rvs).toarray(), device=device, dtype=torch.float32)
        C_0 = experiment(alpha_param, beta_param, gamma_param, C, theta, X, exp_iter=args_global.iterations)
        L = theta.clone().detach().to(device).float()  # Fix for the warning
        acc, std = get_accuracy(C_0, L)
    except Exception as e:
        print(f"Trial failed: {e}")
        return None, None
    end_time = time.time()  # End time for the test case
    elapsed_time = end_time - start_time  # Calculate elapsed time
    estimated_remaining_time_in_days = max(0, remaining_trials - index - 1) * elapsed_time / (24 * 60 * 60)
    print("\n------\t",f"Time taken: {elapsed_time:.2f} seconds","\t------")
    print(f"Using device: {device}; estimated remaining time: {estimated_remaining_time_in_days:.3f} days")
    return acc.item(), std.item()

alpha_param  = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
beta_param   = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
gamma_param  = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
# Three-parameter grid (7^3 = 343), since lambda/Y regularization is removed.
not_visited_set = np.array(np.meshgrid(alpha_param, beta_param, gamma_param)).T.reshape(-1, 3)
not_visited_set = set(map(tuple, not_visited_set))

visited_set = pd.read_csv(filepath)
visited_set = visited_set[['alpha_param', 'beta_param', 'gamma_param']].values
visited_set = set(map(tuple, visited_set))

not_visited_set = sorted(not_visited_set - visited_set)
if args_global.max_trials is not None:
    not_visited_set = not_visited_set[:args_global.max_trials]
remaining_trials = len(not_visited_set)
highest_acc = 0

for index, s in enumerate(not_visited_set):
    test_case_no=index + 1
    print("\n------\t", dataset_global, "\t", r_global, "\t", test_case_no , "|", len(not_visited_set), "\t------")
    alpha_param, beta_param, gamma_param = s
    try:
        acc, std = fitness_function(alpha_param, beta_param, gamma_param)
        if acc is not None:
            highest_acc = max(highest_acc, acc)
            # Store every tested setting so the sweep can be resumed accurately.
            save_readings(alpha_param, beta_param, gamma_param, r_global, k_global, acc, std)
    except Exception as e:
        print(e)
    
print(f"highest accuracy in this run is: {highest_acc}")
