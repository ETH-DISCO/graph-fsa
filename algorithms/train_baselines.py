import time
import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import trange
from sklearn.metrics import accuracy_score

from helpers.mappings import map_x_to_state


import sys, os
models_dir = os.path.abspath('../')
sys.path.append(models_dir)
from models.RecGNN import RecGNN
from models.GNCA import GNNCASimple
from models.GraphChef import RecurrentGraphChef
import pandas as pd


from dataset_generator import distance
from dataset_generator import prefixsum
from dataset_generator import rootvalue
from dataset_generator import pathfinding

from helpers import utils
from helpers import mappings

# following options available "recgnn", "gnca", "graphchef"
DATA_SEED = 1
STATES_N = 2
PATIENCE = 2
INPUT = 4
GNCA_INPUT = 4
model_architecture = "recgnn"
model_save_name = f"{model_architecture}"

# CE Loss relevant to some architectures
ce_loss = torch.nn.CrossEntropyLoss()
bce_loss = torch.nn.BCELoss()


def compute_class_weights(loader):
    total_counts = torch.zeros(STATES_N)
    for data in loader:
        counts = torch.bincount(torch.squeeze(data.y, dim=-1).to(torch.long), minlength=STATES_N)
        total_counts += counts
    weights = 1. / total_counts.clamp(min=1)  # avoid division by zero
    normalized_weights = weights / weights.sum()
    #print("normalized weights", normalized_weights)
    return normalized_weights


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(loader, model, optimizer, dataset_name):
    model.train()
    total_loss = 0

    for data in loader:
        optimizer.zero_grad()
        if model_architecture == "gnca":
            input_data = map_x_to_state(data.x, GNCA_INPUT, dataset_name)
            output = model(input_data, data.edge_index)
        else: 
            output = model(data.x, data.edge_index)


        true_y = torch.nn.functional.one_hot(data.y.to(torch.long), num_classes=2)

        if model_architecture == "gnca":
            true_y = torch.nn.functional.one_hot(data.y.to(torch.long), num_classes=GNCA_INPUT)

        loss = ce_loss(torch.squeeze(output, dim=-1), torch.squeeze(true_y, dim=-2).to(torch.float))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
        optimizer.step()

        total_loss += float(loss) * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def test(loader, model, dataset_name, iterations=None, hardmax=False):
    model.eval()
    total_loss = 0
    accuracies = []

    if hardmax:
        model.set_hardmax(True)

    if iterations:
        prev_iterations = model.iterations
        model.set_iterations(iterations)

    for data in loader:

        if model_architecture == "gnca":
            input_data = map_x_to_state(data.x, GNCA_INPUT, dataset_name)
            #print("input_data.x", input_data.sh)
            output = model(input_data, data.edge_index)

            # compute accuracy
            y_pred = torch.argmax(output, dim=1)
            accuracy = accuracy_score(data.y, y_pred)

            # compute outputs

            true_y = torch.nn.functional.one_hot(data.y.to(torch.long), num_classes=GNCA_INPUT)
            loss = ce_loss(torch.squeeze(output, dim=-1), torch.squeeze(true_y, dim=-2).to(torch.float))
        else:

            output = model(data.x, data.edge_index)

            y_pred = torch.argmax(output, dim=1)
            accuracy = accuracy_score(data.y, y_pred)

            true_y = torch.nn.functional.one_hot(data.y.to(torch.long), num_classes=2)
            loss = ce_loss(torch.squeeze(output, dim=-1), torch.squeeze(true_y, dim=-2).to(torch.float))


        total_loss += float(loss) * data.num_graphs
        accuracies.append(accuracy)

    if iterations:
        model.set_iterations(prev_iterations)

    if hardmax:
        model.set_hardmax(False)

    return sum(accuracies) / len(accuracies) if accuracies else 0, total_loss / len(loader.dataset)

def train_model(params):
    global STATES_N, GNCA_INPUT, INPUT
    INPUT = 2
    print(f"----- model: {STATES_N}  -----")
    set_seed(DATA_SEED)
    dataset_name = params["dataset_name"]
    
    if dataset_name == "Distance":
        train_dataset = distance.Distance(
            num_graphs=100,
            num_nodes=4,
        ).data + distance.Distance(
            num_graphs=100,
            num_nodes=10,
        ).data + distance.Distance(
            num_graphs=50,
            num_nodes=20,
        ).data + distance.Distance(
            num_graphs=50,
            num_nodes=30,
        ).data 

        val_dataset = distance.Distance(
            num_graphs=20,
            num_nodes=7,
        ).data

        test_dataset = distance.Distance(
            num_graphs=10,
            num_nodes=20,
        ).data
        INPUT = 2
        GNCA_INPUT = 4
        dataset_function = distance.Distance

    elif dataset_name == "PrefixSum":

        train_dataset = prefixsum.PrefixSum(
            num_graphs=350, rangeValue=5, num_nodes=4
        ).data

        val_dataset = prefixsum.PrefixSum(num_graphs=20, rangeValue=4, num_nodes=3).data

        test_dataset = prefixsum.PrefixSum(
            num_graphs=5,
            num_nodes=10,
        ).data

        dataset_function = prefixsum.PrefixSum
        INPUT = 4
        GNCA_INPUT = 6
    elif dataset_name == "RootValue":

        train_dataset = rootvalue.RootValue(
            num_nodes=4, num_graphs=50, rangeValue=4
        ).data

        val_dataset = rootvalue.RootValue(num_nodes=4, num_graphs=50, rangeValue=4).data

        test_dataset = rootvalue.RootValue(
            num_nodes=5,
            rangeValue=2,
        ).data
        INPUT = 3
        GNCA_INPUT = 5

        dataset_function = rootvalue.RootValue

    elif dataset_name == "PathFinding":

        train_dataset = pathfinding.PathFinding(
            num_nodes=4, num_graphs=120, rangeValue=3
        ).data

        val_dataset = pathfinding.PathFinding(
            num_nodes=3, num_graphs=20, rangeValue=2
        ).data

        test_dataset = pathfinding.PathFinding(
            num_nodes=20,
            num_graphs=5,
        ).data
        INPUT = 2
        GNCA_INPUT = 5
        dataset_function = pathfinding.PathFinding

    STATES_N = len(train_dataset[0].x)

    dataset = train_dataset

    test_datasets = []
    tests = [(4, 6), (10, 20), (20, 30), (50, 75), (100, 150)]
    for (nodes, iterations) in tests:
    # extra tests for my graph
        test_dataset = dataset_function(num_graphs=100,num_nodes=nodes).data
        test_datasets.append(DataLoader(test_dataset, batch_size=1)
)



    set_seed(params["seed"])
    # ToDO: adjust iterations based on problem
    iterations = 25

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=1)

    if model_architecture == "gnca":
        model = GNNCASimple(hidden=256, activation=torch.sigmoid, dim_size=GNCA_INPUT)
    
    elif model_architecture == "graphchef":
        model = RecurrentGraphChef(INPUT, 2, bounding_parameter=2,state_size=STATES_N, gumbel=True, use_pooling=False, use_edge_mlp=False, network="mlp")
    
    else:
        model = RecGNN(INPUT, 16, 2, 4, 0.05, skip_input=True, conv="gru-mlp", aggregation="add")



    ce_loss = torch.nn.CrossEntropyLoss()
    bce_loss = torch.nn.BCELoss()
    

    model.set_iterations(8)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=PATIENCE, min_lr=0.00001)

    best_val_loss = np.inf
    pbar = trange(1, params["epochs"] + 1)
    
    for epoch in pbar:
        loss = train(train_loader, model, optimizer, dataset_name)
        train_acc, _ = test(train_loader, model, dataset_name)
        val_acc, val_loss = test(val_loader, model, dataset_name)
        test_acc, _ = test(test_loader, model, dataset_name)

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, f"runs/{dataset_name}/{model_save_name}.pt")
        
        pbar.set_description(f"Epoch: {epoch:04d}, Loss: {loss:.3f}, Train: {train_acc:.3f}, Val: {val_acc:.3f}, Test: {test_acc:.3f}")
        print("\n")
    
    print("---------- GNN (Acc, Loss) ----------")
    model = torch.load(f"runs/{dataset_name}/{model_save_name}.pt")

    train_acc, val_acc = test(train_loader, model, dataset_name)[0], test(val_loader, model, dataset_name)[0]
    print(f"Train: {train_acc}, Val: {val_acc}")

    #for i,test_loader in enumerate(test_datasets):
    #    model.set_iterations(tests[i][1])
    #    test_acc = test(test_loader, model, dataset_name)[0]
    #    print(f"Extrapolation Softmax {test_acc:.3f} \n \n")

    #print("---------- GNN-FSM HARDMAX ----------")
    #for i,test_loader in enumerate(test_datasets):
    #    model.set_iterations(tests[i][1])
    #    test_acc = test(test_loader, model, dataset_name, iterations=iterations, hardmax=True)[0]
    #    print(f"n = {nodes} {test_acc}")

    rows = []

    rows.append([model_architecture, '1', params['seed'], val_acc, params['dataset_name']])

    for i,test_loader in enumerate(test_datasets):
        model.set_iterations(tests[i][1])
        test_acc = test(test_loader, model, dataset_name)[0]
        rows.append([model_architecture, tests[i][0], params['seed'], test_acc, params['dataset_name']])

        print(f"n = {nodes} {test_acc}")

    df = pd.DataFrame(rows, columns=['model', 'graph_size', 'seed', 'test_acc', 'dataset'])

    dataset_name = params['dataset_name']
    seed = params['seed']
    df.to_csv(f'results/{model_architecture}_{dataset_name}_{seed}.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate 1D CCA")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--seed", default=42, type=int)  # add seed argument

    parser.add_argument(
        "--dataset",
        default="Distance",
        help="Name of the dataset to run the experiment on",
    )


    parser.add_argument("--model", default="recgnn", type=str)
    parser.add_argument("--patience", default=PATIENCE, type=int)


    args = parser.parse_args()

    PATIENCE = args.patience
    model_architecture = args.model

    params = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "dataset_name": args.dataset,
    }

    set_seed(args.seed)
    train_model(params)
