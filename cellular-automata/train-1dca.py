import time
import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import trange
from sklearn.metrics import accuracy_score
import json
import sys, os
import pandas as pd
models_dir = os.path.abspath('../')
sys.path.append(models_dir)

from models.GraphFSANeighbourhood import NeuralCA
from models.RecGNN import RecGNN
from models.GNCA import GNNCASimple
from models.GraphChef import RecurrentGraphChef

from helpers import utils
from helpers import mappings
from datasets.gameoflife import GameOfLifeGraph
from datasets.onedimensionalca import OneDimensionalAutomataGridGraph


STATES_N = 2
RULE_NUMBER = 42
GRID_SIZE = 4
STEPS = 2

# TRAINING PARAMETERS
PATIENCE = 2
CLIP = False
# following options available "graphfsa", "recgnn", "gnca", "graphchef"
model_architecture = "graphfsa"
model_save_name = f"ca_rule_{RULE_NUMBER}_{model_architecture}"

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

def train(loader, model, optimizer, device):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        # use loss depending in model architecture 
        if model_architecture == "graphfsa":
            
            # apply input_state mapping for GraphFSA architecture
            input_states = mappings.map_x_to_state(data.x, STATES_N)
            output = model(input_states, data.edge_index)
        
            output_y = torch.nn.functional.one_hot(data.y.to(torch.long), num_classes=STATES_N)
            output_y = torch.squeeze(torch.squeeze(output_y, dim=-2))

            loss = torch.square(output - output_y).sum()

            #finalStateLoss = utils.FinalStateLoss()
            #additiveLoss = finalStateLoss(model.T, [0, 1])
            #loss += additiveLoss

        elif model_architecture == "gnca":
            output = model(data.x, data.edge_index)
            output = torch.squeeze(output, dim=-1)
            data.y = torch.squeeze(data.y, dim=-1)

            loss = bce_loss(output, data.y)

        else:
            output = model(data.x, data.edge_index)
            true_y = torch.nn.functional.one_hot(data.y.to(torch.long), num_classes=2)
            loss = ce_loss(torch.squeeze(output, dim=-1), torch.squeeze(true_y, dim=-2).to(torch.float))


        loss.backward()
        if CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Example maximum gradient norm
        optimizer.step()

        total_loss += float(loss) * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def test(loader, model, iterations=None, hardmax=False):
    model.eval()
    total_loss = 0
    accuracies = []

    if hardmax:
        model.set_hardmax(True)

    if iterations:
        prev_iterations = model.iterations
        model.set_iterations(iterations)

    for data in loader:

        if model_architecture == "graphfsa":
            input_states = mappings.map_x_to_state(data.x, STATES_N)

            output = model(input_states, data.edge_index)
            
            y_pred = torch.argmax(output, dim=1)
            accuracy = accuracy_score(data.y, y_pred)

            output_y = torch.nn.functional.one_hot(data.y.to(torch.long), num_classes=STATES_N)
            output_y = torch.squeeze(torch.squeeze(output_y, dim=-2))

            loss = torch.square(output - output_y).sum()
            # alternative loss function is we only consider the maximum (strange?)
            #loss = torch.square(data.y - y_pred).sum()

        elif model_architecture == "gnca":
            output = model(data.x, data.edge_index)

            # compute accuracy
            y_pred = torch.argmax(output, dim=1)
            y_pred = (output >= 0.5).to(torch.long)
            accuracy = accuracy_score(data.y, y_pred)

            # compute outputs
            output = torch.squeeze(output, dim=-1)
            data.y = torch.squeeze(data.y, dim=-1)

            loss = bce_loss(output, data.y)

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
    print(f"----- model: {STATES_N}  -----")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = str(time.time_ns())

    train_dataset = OneDimensionalAutomataGridGraph(GRID_SIZE, 2500, rule_number=RULE_NUMBER, steps=STEPS, seed=42).data
    val_dataset = OneDimensionalAutomataGridGraph(GRID_SIZE, 10, rule_number=RULE_NUMBER, steps=STEPS, seed=42+1).data
    test_dataset = OneDimensionalAutomataGridGraph(GRID_SIZE, 10, rule_number=RULE_NUMBER, steps=STEPS, seed=42+2).data

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"])

    set_seed(args.seed)

    # use class weights when using the ce_loss
    weights = compute_class_weights(train_loader)
    global ce_loss, bce_loss

    ce_loss = torch.nn.CrossEntropyLoss(weight=weights)
    bce_loss = torch.nn.BCELoss()

    if model_architecture == "graphfsa":
        model = NeuralCA(3 ** STATES_N, STATES_N, GRID_SIZE)
    
    elif model_architecture == "gnca":
        model = GNNCASimple(hidden=256, activation=torch.sigmoid)
    
    elif model_architecture == "graphchef":
        model = RecurrentGraphChef(1, 2, bounding_parameter=2,state_size=STATES_N, gumbel=True, use_edge_mlp=False, network="mlp")
    
    else:
        model = RecGNN(1, 16, 2, 4, 0.05, skip_input=True, conv="gru-mlp", aggregation="add").to(device)

    model.set_iterations(STEPS)

    if model_architecture == "graphfsa":
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], betas=(0.5, 0.5))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=PATIENCE, min_lr=0.00001)

    best_val_loss = np.inf
    pbar = trange(1, params["epochs"] + 1)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters:", total_params)

    if not os.path.isdir("runs/ca"):
        os.mkdir("runs/ca")

    for epoch in pbar:
        loss = train(train_loader, model, optimizer, device)
        train_acc, _ = test(train_loader, model)
        val_acc, val_loss = test(val_loader, model)
        test_acc, _ = test(test_loader, model)

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, f"runs/ca/{model_save_name}_{timestamp}.pt")
        
        pbar.set_description(f"Epoch: {epoch:04d}, Loss: {loss:.3f}, Train: {train_acc:.3f}, Val: {val_acc:.3f}, Test: {test_acc:.3f}")
        print("\n")
    
    print("---------- GNN (Acc, Loss) ----------")
    device = torch.device("cpu")
    model = torch.load(f"runs/ca/{model_save_name}_{timestamp}.pt")

    train_acc, val_acc = test(train_loader, model)[0], test(val_loader, model)[0]
    print(f"Train: {train_acc}, Val: {val_acc}")

    test_acc = test(test_loader, model)[0]
    print(f"Extrapolation Softmax {test_acc:.3f} \n \n")

    print("---------- GNN-FSM HARDMAX ----------")

    rows = []

    for num_iters in ([1,2,3,4] + list(range(5,101, 5))):
        test_dataset = OneDimensionalAutomataGridGraph(grid_size=10, num_graphs=100, rule_number=RULE_NUMBER, steps=num_iters, seed=100).data

        test_loader = DataLoader(test_dataset, batch_size=1)

        model.set_iterations(num_iters)
        if model_architecture == "graphfsa":
            model.set_grid_size(10)
            hardmax = True
        else:
            hardmax = False

        test_acc = test(test_loader, model, iterations=num_iters, hardmax=hardmax)[0]
        rows.append([model_architecture, num_iters, params['seed'], test_acc])
        print(f"n = {num_iters} {test_acc}")

    df = pd.DataFrame(rows, columns=['model', 'num_iters', 'seed', 'test_acc'])
    seed = params['seed']
    df.to_csv(f'results_1dca/{model_architecture}_{seed}.csv')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate 1D CCA")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--seed", default=41, type=int)  # add seed argument

    parser.add_argument("--states_n", default=STATES_N, type=int)
    parser.add_argument("--grid_size", default=GRID_SIZE, type=int)
    parser.add_argument("--steps", default=STEPS, type=int)
    parser.add_argument("--model", default=model_architecture, type=str)
    parser.add_argument("--patience", default=PATIENCE, type=int)
    parser.add_argument("--clip", default=CLIP, type=bool)
    parser.add_argument("--rule_number", default=RULE_NUMBER, type=int)

    args = parser.parse_args()
    args = parser.parse_args()
    STATES_N = args.states_n
    GRID_SIZE = args.grid_size
    RULE_NUMBER = args.rule_number
    STEPS = args.steps
    model_architecture = args.model
    PATIENCE = args.patience
    CLIP = args.clip

    model_save_name = f"gol_steps_{STEPS}_model_{model_architecture}"
    print(" run with ", args)

    params = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed
    }

    set_seed(args.seed)
    train_model(params)
