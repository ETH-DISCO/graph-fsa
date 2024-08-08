import numpy as np
import argparse
import torch
import pandas as pd

from fsm_dataset import dataset as dataset_generator


import sys, os
models_dir = os.path.abspath('../')
sys.path.append(models_dir)

from models.RecGNN import RecGNN
from models.GNCA import GNNCASimple
from models.GraphChef import RecurrentGraphChef

from helpers import utils
import os

from torch_geometric.loader import DataLoader


from tqdm import trange

#from state_machine_generator import generator

from sklearn.metrics import accuracy_score

import random
import time
import json


# options ["recgnn", "gnca", "graphchef"]
MODEL_NAME = "recgnn"
PATIENCE = 2
DATA_SEED = 42
CLIP = False
STEPS = 5

def train_model(
    params,
):

    # model parameters
    
    print(f"----- model: {MODEL_NAME}- seed: {params['seed']} -----")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = params["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    now = str(time.time())

    model_dir_name = f"{MODEL_NAME}_seed_" + str(params["seed"]) + "_" + now

    model_path = os.path.join("runs/spase-1", model_dir_name)
    model_file_path = os.path.join(model_path, "model.pt")
    result_file_path = os.path.join(model_path, "run_result.json")

    if not os.path.exists(model_path):
        # If it doesn't exist, create it
        os.makedirs(model_path)

    # details about fsm to generate
    starting_states = [2,3]
    final_states = [0, 1]
    num_states = 4

    dataset = dataset_generator.FSMDataset("datasets/fsm_dataset/", dataset_generator.GRAPH_TYPE.TREE, num_states, starting_states, final_states, seed=DATA_SEED, sparse=False)

    #model = GCN(num_states, 4)
    if MODEL_NAME.lower() == "recgnn":
        model = RecGNN(num_states, 16, len(final_states), 4, 0.1, skip_input=True, conv="gru-mlp", aggregation="add").to(device)
    elif MODEL_NAME.lower() == "graphchef":
        model = RecurrentGraphChef(num_states, len(final_states), bounding_parameter=1, state_size=num_states, gumbel=True, network="mlp")
    else:
        model = GNNCASimple(hidden=256, dim_size=num_states, activation=torch.sigmoid)


    train_dataset = dataset.data
    val_dataset = dataset.val_data
    test_dataset = dataset.test_data 
    test_ood_dataset = dataset.ood_test_data

    dataset = train_dataset

    # ToDO: adjust iterations based on problem
    model.set_iterations(5)

    # Data Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )

    val_loader = DataLoader(val_dataset, batch_size=1)
    test_loaders = [DataLoader(t, batch_size=1) for t in test_dataset]
    test_ood_loader = DataLoader(test_ood_dataset, batch_size=params["batch_size"])

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["learning_rate"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=PATIENCE, min_lr=0.00001
    )

    ce_loss = torch.nn.CrossEntropyLoss()

    if params["weighted_loss"]:
        class_weights = utils.compute_class_weights2(train_loader, 4)
        ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)

    def train(loader):
        model.train()
        total_loss = 0

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()

            output = model(
                data.x, data.edge_index
            )
            #output_true = torch.argmax(output, dim=-1) # .to(torch.long)
            #output = torch.argmax(output, dim=-1)
            data_y_true = torch.argmax(data.y, dim=-1).to(torch.long)
            loss = ce_loss(output, data_y_true)
                
            #l2_norm = torch.mean(torch.linalg.norm(states[-1], dim=1))
            #loss += 0.0001 * l2_norm

            loss.backward()
            if CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Example maximum gradient norm        
            optimizer.step()

            total_loss += float(loss) * data.num_graphs
        return total_loss / len(loader.dataset)


    @torch.no_grad()
    def test(loader, iterations=None, hardmax=False, log=False, auto_adjust=False):
        model.eval()
        total_loss = 0
        accuracies = []
        count = 0

        accuracy_dict = {}


        ce_loss = torch.nn.CrossEntropyLoss()

        if iterations:
            prev_iterations = model.iterations
            model.set_iterations(iterations)

        for data in loader:
            output = model(data.x, data.edge_index)

            if auto_adjust:
                model.set_iterations(len(data.x) + 5)

            output = output[data.mask.to(torch.bool)]
            data.y = data.y[data.mask.to(torch.bool)]

            y_pred = torch.argmax(output[:, : len(final_states)], dim=-1)
            y_true = torch.argmax(data.y[:, : len(final_states)], dim=-1)

            accuracy = accuracy_score(y_true, y_pred)

            if data.x.shape[0] not in accuracy_dict.keys():
                accuracy_dict[data.x.shape[0]] = [accuracy]
            else: 
                accuracy_dict[data.x.shape[0]].append(accuracy)

            count += 1

            output_y = data.y.to(torch.long)[:, :len(final_states)]


            data_y_true = torch.argmax(data.y, dim=-1).to(torch.long)

            loss = ce_loss(output, data_y_true)

            total_loss += float(loss) * data.num_graphs

            accuracies.append(accuracy)

        if iterations:
            model.set_iterations(prev_iterations)

        for key in accuracy_dict.keys():
            accuracy_dict[key] =  sum(accuracy_dict[key])/len(accuracy_dict[key])


        if log:
            print("accuracy_dict", accuracy_dict)

        if len(accuracies) == 0 or len(loader.dataset) == 0:
            return 0, 0

        return sum(accuracies) / len(accuracies), total_loss / len(loader.dataset)

    best_val_loss = np.inf
    best_val_acc = 0
    pbar = trange(1, params["epochs"] + 1)

    count = 0


    for epoch in pbar:
        # in every epoch we randomly choose a diameter between None, 1 - 7 and adjust the number of iterations

        loss = train(train_loader)

        count += 1

        train_acc, _ = test(train_loader, log=False)
        val_acc, val_loss = test(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc

            torch.save(model, model_file_path)
        print("\n")
        pbar.set_description(
            f"Epoch: {epoch:04d}, Loss: {loss:.3f} Train: {train_acc:.3f},"
            f" Val: {val_acc:.3f}"
            f" Best Val Acc.:  {best_val_acc:.3f}",
        )


    print("---------- GNN (Acc, Loss) ----------")
    device = torch.device("cpu")
    model = torch.load(model_file_path).to(device)

    # accuracies of best model
    train_acc = test(train_loader)[0]
    val_acc = test(val_loader)[0]

    print(f"Train: {train_acc}, Val: {val_acc}")


    test_accs = []

    for test_loader in test_loaders:
        test_acc = test(test_loader, log=True, auto_adjust=True)[0]
        print(f"Extrapolation Accuracy {test_acc} \n \n")
        test_accs.append(test_acc)

    test_ood_acc = test(test_ood_loader, auto_adjust=True)[0]
    print(f"OOD Accuracy {test_ood_acc:.3f} \n \n")

    result_object = {   
        "train_acc": train_acc, 
        "val_acc": val_acc,
        "extrapolation_acc": test_accs,
        "ood_acc": test_ood_acc, 
        "loss": loss, 
        "seed": params["seed"],
        "starting_states": starting_states,
        "final_states": final_states,
        "num_states_true": num_states
    }

    sizes = [10, 20, 50, 100]

    rows = []

    rows.append([MODEL_NAME, '1', params['seed'], val_acc])

    for i in range(len(test_accs)):
        rows.append([MODEL_NAME, sizes[i], params['seed'], test_accs[i]])

    df = pd.DataFrame(rows, columns=['model', 'graph_size', 'seed', 'test_acc'])
    seed = params['seed']

    df.to_csv(f'results/{MODEL_NAME}_{seed}.csv')

    with open(result_file_path, "w") as f:
        f.write(json.dumps(result_object))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the State Machine GNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--model", default="recgnn", type=str)
    parser.add_argument("--patience", default=PATIENCE, type=int)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--clip", default=CLIP, type=bool)
    parser.add_argument("--steps", default=STEPS, type=int)
    parser.add_argument("--weighted_loss", default=False, type=bool)


    args = parser.parse_args()

    MODEL_NAME = args.model
    PATIENCE = args.patience
    CLIP = args.clip
    STEPS = args.steps

    print("args", args)

    params = {
        "learning_rate": args.learning_rate,#0.001,  #  0.0004,  # 0.01,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "weighted_loss": args.weighted_loss,
        "seed": args.seed,
    }

    train_model(
        params,
    )
