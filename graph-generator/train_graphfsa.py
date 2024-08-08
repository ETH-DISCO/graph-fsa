import argparse
import json
import os
import random
import time
from tqdm import trange
import numpy as np
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score

from fsm_dataset import dataset as dataset_generator

import sys, os
models_dir = os.path.abspath('../')
sys.path.append(models_dir)
from models.GraphFSA import GraphFSA

from helpers import utils

from torch_geometric.loader import DataLoader
#from state_machine_generator import generator

from sklearn.metrics import accuracy_score


PATIENCE = 2
DATA_SEED = 42

def train_model(params):
    """
    Train the model with the given parameters.
    """

    # set seed for the model - currently we use same seed for dataset generator 
    seed = params["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ToDO: adjust iterations based on problem
    iterations = 15

    # model parameters
    STATES_N = params["num_states"]

    print(f"----- model: {STATES_N} {params['seed']} -----")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model directory and file paths
    now = str(time.time())
    model_dir_name = f"state_{STATES_N}_seed_{params['seed']}_{now}"
    model_path = os.path.join("runs/sparse-1", model_dir_name)
    os.makedirs(model_path, exist_ok=True)

    model_file_path = os.path.join(model_path, "model.pt")
    result_file_path = os.path.join(model_path, "run_result.json")

    fsm_visual_learned_path = os.path.join(model_path, "fsm_learned")
    fsm_visual_used_path = os.path.join(model_path, "fsm_transitions_used")

    if not os.path.exists(model_path):
        # If it doesn't exist, create it
        os.makedirs(model_path)

    # details about fsm to generate
    starting_states = [2,3]
    final_states = [0,1]
    num_states = 4
    
    dataset = dataset_generator.FSMDataset("datasets/fsm_dataset/", dataset_generator.GRAPH_TYPE.TREE, num_states, starting_states, final_states, seed=DATA_SEED, sparse=False)

    train_dataset = dataset.data
    val_dataset = dataset.val_data
    test_dataset = dataset.test_data 
    test_ood_dataset = dataset.ood_test_data 

    dataset = train_dataset

    # Data Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )

    val_loader = DataLoader(val_dataset, batch_size=1)
    test_loaders = [DataLoader(t, batch_size=1) for t in test_dataset]

    model = GraphFSA(2, STATES_N)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["learning_rate"], betas=(0.5, 0.5)
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=PATIENCE, min_lr=0.00001
    )

    if params["weighted_loss"]:
        class_weights = utils.compute_class_weights(train_loader)

    def map_extended(value):
        extended_value = torch.zeros((value.shape[0], STATES_N))

        for i, row in enumerate(value):
            for j in range(len(row)):

                if j >= STATES_N:
                    continue

                extended_value[i][j] = row[j]

        return extended_value

    def train(loader):
        model.train()
        total_loss = 0

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()

            # randomly add 0 or 1 to the number of iterations
            curr_iterations = model.iterations
            if params["iteration_offset"]:
                model.set_iterations(
                    curr_iterations + random.randint(0, params["iteration_offset_size"])
                )

            input_data = map_extended(data.x)

            model.set_logging(True)
            output, states, _ = model(input_data, data.edge_index)
            model.set_logging(False)

            # loss = ce_loss(output, data.y.to(torch.long))
            # extend output and use squared distance loss

            # add entropy loss to reduce number of states
            # entropy_loss = utils.HLoss()
            finalStateLoss = utils.FinalStateLoss()
            startingStateLoss = utils.NoOtherStartingStatesLoss()

            additiveLoss = 0
            additiveLoss = finalStateLoss(model.T, [0, 1])
            # with the current state machines generated we don't want this loss
            # extraLoss = startingStateLoss(model.T, [2, 3])

            # output_y = torch.nn.functional.one_hot(
            #    data.y.to(torch.long), num_classes=STATES_N
            # )

            output_y = map_extended(data.y).to(torch.long)

            if params["weighted_loss"]:
                size = output.shape[0]
                weight = torch.zeros(size)

                for i in range(output.shape[0]):
                    if data.y[i] == 1:
                        weight[i] = class_weights[size][1]
                    else:
                        weight[i] = class_weights[size][0]

                loss_per_node = torch.square(output - output_y).sum(dim=-1) * weight
                loss = loss_per_node.sum()
            else:
                loss = torch.square(output - output_y).sum()

            loss += 1 * additiveLoss

            loss.backward()
            optimizer.step()

            model.set_iterations(curr_iterations)

            total_loss += float(loss) * data.num_graphs
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def test(loader, iterations=None, hardmax=False, auto_adjust=False, logging=False):
        model.eval()
        total_loss = 0
        accuracies = []
        count = 0

        if hardmax:
            model.set_hardmax(True)

        if logging:
            model.set_logging(True)
            all_outputs = []
            all_states = []
            all_transitions = []

        if iterations:
            prev_iterations = model.iterations
            model.set_iterations(iterations)

        for data in loader:
            input_data = map_extended(data.x)
            if logging:
                output, states, transitions = model(input_data, data.edge_index)

                # add outputs, states, transitions
                all_outputs.append(output)
                all_states.append(states)
                all_transitions.append(transitions)

            else:
                output = model(input_data, data.edge_index)

            if auto_adjust:
                model.set_iterations(len(data.x) + 5)

            output = output[data.mask.to(torch.bool)]
            data.y = data.y[data.mask.to(torch.bool)]

            y_pred = torch.argmax(output[:, : len(final_states)], dim=-1)
            y_true = torch.argmax(data.y[:, : len(final_states)], dim=-1)

            accuracy = accuracy_score(y_true, y_pred)

            count += 1

            output_y = map_extended(data.y).to(torch.long)
            loss = torch.square(output - output_y).sum()

            total_loss += float(loss) * data.num_graphs

            accuracies.append(accuracy)

        if iterations:
            model.set_iterations(prev_iterations)

        if hardmax:
            model.set_hardmax(False)
        
        if logging:
            model.set_logging(False)
            return sum(accuracies) / len(accuracies), total_loss / len(loader.dataset), all_outputs, all_states, all_transitions
        
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

        train_acc, _ = test(train_loader)
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
    train_acc = test(train_loader, hardmax=True)[0]
    val_acc = test(val_loader, hardmax=True)[0]

    print(f"Train: {train_acc}, Val: {val_acc}")

    for test_loader in test_loaders:
        test_acc = test(test_loader, iterations=iterations)[0]

    #transition_matrix_new = T = torch.nn.functional.one_hot(
    #    torch.argmax(model.T, dim=-1), model.T.shape[-1]
    #).to(torch.float32).numpy()

    #learnedFSM = dataset_generator.FSM(STATES_N, starting_states, final_states, transition_matrix_new) #, neighbourhood_aggregation=True)
    #learnedFSM.visualize(fsm_visual_learned_path)

        print(f"Extrapolation Softmax {test_acc:.3f} \n \n")


    test_accs = []
    print("---------- GNN-FSM HARDMAX ----------")
    for test_loader in test_loaders:
        return_values = test(test_loader, hardmax=True, auto_adjust=True, logging=True)
        test_1_acc = return_values[0]
        print(f"Extrapolation Hardmax {test_1_acc:.3f}")
        test_accs.append(test_1_acc)
        all_outputs, all_states, all_transitions = return_values[2], return_values[3], return_values[4]

    #learnedFSM.visualize_used_transitions(fsm_visual_used_path, all_outputs, all_states, all_transitions)

    test_ood_loader = DataLoader(test_ood_dataset, batch_size=params["batch_size"])
    test_ood_acc = test(test_ood_loader, hardmax=True, auto_adjust=True)[0]

    result_object = {   
        "train_acc": train_acc, 
        "val_acc": val_acc,
        "extrapolation_acc": test_accs,
        "ood_acc": test_ood_acc, 
        "loss": loss, 
        "seed": params["seed"],
        "starting_states": starting_states,
        "final_states": final_states,
        "num_states_true": num_states,
        "num_states_model": STATES_N,
    }

    sizes = [10, 20, 50, 100]

    rows = []

    rows.append(['graphfsa', '1', params['seed'], val_acc])

    for i in range(len(test_accs)):
        rows.append(['graphfsa', sizes[i], params['seed'], test_accs[i]])

    df = pd.DataFrame(rows, columns=['model', 'graph_size', 'seed', 'test_acc'])
    seed = params['seed']

    df.to_csv(f'results/graphfsa_{seed}.csv')


    with open(result_file_path, "w") as f:
        f.write(json.dumps(result_object))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the State Machine GNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--states", default=4, type=int)

    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--patience", default=PATIENCE, type=int)

    args = parser.parse_args()
    PATIENCE=args.patience

    params = {
        "learning_rate": args.learning_rate,#0.25,  #  0.0004,  # 0.01,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "iteration_offset": True,
        "iteration_offset_size": 1,
        "weighted_loss": False,
        "seed": args.seed,
        "num_states": args.states, 
    }

    train_model(
        params,
    )
