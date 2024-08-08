import numpy as np
import argparse
import torch


import sys, os
models_dir = os.path.abspath('../')
sys.path.append(models_dir)

from models.GraphFSA import GraphFSA

from helpers import utils
from helpers import mappings
import pandas as pd

from torch_geometric.loader import DataLoader

from tqdm import trange
import time

from dataset_generator import distance
from dataset_generator import prefixsum
from dataset_generator import rootvalue
from dataset_generator import pathfinding

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from torch.utils.tensorboard import SummaryWriter
import random

DATA_SEED = 1
GLOBAL_COUNT = 0
# model parameters
STATES_N = 4
PATIENCE=2

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    dataset_name,
    params,
):

    global STATES_N

    print(
        f"----- {dataset_name} model: {STATES_N} memory: {params['transition_memory']} -----"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = str(time.time())

    dataset_function = None

    set_seed(DATA_SEED)

    if dataset_name == "Distance":
        train_dataset = distance.Distance(
            num_graphs=300,
            num_nodes=6
        ).data +  distance.Distance(
            num_graphs=100,
            num_nodes=8
        ).data +  distance.Distance(
            num_graphs=100,
            num_nodes=4
        ).data + distance.Distance(
            num_graphs=20,
            num_nodes=1
        ).data + distance.Distance(
            num_graphs=20,
            num_nodes=2
        ).data 

        val_dataset = distance.Distance(
            num_graphs=20,
            num_nodes=7,
        ).data

        test_dataset = distance.Distance(
            num_graphs=10,
            num_nodes=20,
        ).data
        STATES_N = 4
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
        STATES_N = 6
        dataset_function = prefixsum.PrefixSum

    elif dataset_name == "RootValue":

        train_dataset = rootvalue.RootValue(
            num_nodes=4, num_graphs=50, rangeValue=4
        ).data

        val_dataset = rootvalue.RootValue(num_nodes=4, num_graphs=50, rangeValue=4).data

        test_dataset = rootvalue.RootValue(
            num_nodes=5,
            rangeValue=2,
        ).data
        STATES_N = 5
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
        STATES_N = 5
        dataset_function = pathfinding.PathFinding

    dataset = train_dataset



    # mapping of tests with (number nodes, number_iterations)
    tests = [(4, 6), (10, 20), (20, 30), (50, 75), (100, 150)]
    test_loaders = []
    for (nodes, iterations) in tests:

        # extra tests for my graph
        test_dataset = dataset_function(
            num_graphs=100,
            num_nodes=nodes,
        ).data
        test_loader = DataLoader(test_dataset, batch_size=1)
        test_loaders.append(test_loader)

    set_seed(params["seed"])
    # ToDO: adjust iterations based on problem
    iterations = 12

    # Data Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )

    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"])

    model = GraphFSA(1, STATES_N)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["learning_rate"], betas=(0.5, 0.5)
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=PATIENCE, min_lr=0.00001
    )

    ce_loss = torch.nn.CrossEntropyLoss()

    if params["weighted_loss"]:
        class_weights = utils.compute_class_weights(train_loader)

    def train(loader):
        model.train()
        total_loss = 0

        count = 0

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()

            # randomly add 0 or 1 to the number of iterations
            curr_iterations = model.iterations
            if params["iteration_offset"]:
                model.set_iterations(
                    curr_iterations + random.randint(0, params["iteration_offset_size"])
                )
            input_states = mappings.map_x_to_state(data.x, STATES_N, dataset_name)

            model.set_logging(True)
            output, states, _ = model(input_states, data.edge_index)
            model.set_logging(False)

            global GLOBAL_COUNT

            # loss = ce_loss(output, data.y.to(torch.long))
            # extend output and use squared distance loss

            # add entropy loss to reduce number of states
            # entropy_loss = utils.HLoss()
            finalStateLoss = utils.FinalStateLoss()
            startingStateLoss = utils.NoOtherStartingStatesLoss()

            additiveLoss = 0
            if params["multi_machine"]:
                for T in model.Ts:
                    additiveLoss += finalStateLoss(T, [0, 1])
            else:
                additiveLoss = finalStateLoss(model.T, [0, 1])
            #   extraLoss = startingStateLoss(model.T, [2, 3])

            output_y = torch.nn.functional.one_hot(
                data.y.to(torch.long), num_classes=STATES_N
            )

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

            if params["additive_weight_factor"]:
                additive_weight_factor = max(0, GLOBAL_COUNT - 5) / 100
            else:
                additive_weight_factor = 1

            loss += (
                additive_weight_factor * additiveLoss
            )  # + 0.001 * entropy_loss(states)

            # loss += extraLoss

            loss.backward()
            optimizer.step()

            GLOBAL_COUNT += 1

            model.set_iterations(curr_iterations)

            total_loss += float(loss) * data.num_graphs
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def test(loader, iterations=None, hardmax=False):
        model.eval()
        total_loss = 0
        accuracies = []

        if hardmax:
            model.set_hardmax(True)

        if iterations:
            prev_iterations = model.iterations
            model.set_iterations(iterations)

        for data in loader:

            input_states = mappings.map_x_to_state(
                data.x, STATES_N, dataset_name, use_noise=(not hardmax)
            )

            if params["multi_machine"]:
                start_ids = mappings.map_x_to_start_state_id(data.x)
                output = model(input_states, data.edge_index, start_ids)

            elif params["transition_memory"]:
                input_memory = mappings.map_x_to_memory(data.x)
                output = model(input_states, data.edge_index, input_memory)
            else:
                output = model(input_states, data.edge_index)

            y_pred = torch.argmax(output[:, :2], dim=1)

            #accuracy = accuracy_score(data.y, y_pred)
            accuracy = accuracy_score(data.y, y_pred)
            loss = ce_loss(output, data.y.to(torch.long))

            total_loss += float(loss) * data.num_graphs

            accuracies.append(accuracy)

        if iterations:
            model.set_iterations(prev_iterations)

        if hardmax:
            model.set_hardmax(False)

        if len(accuracies) == 0 or len(loader.dataset) == 0:
            return 0, 0

        return sum(accuracies) / len(accuracies), total_loss / len(loader.dataset)

    model_save_name = (
        dataset_name + "_" + str(STATES_N) + "_mem_" + str(params["transition_memory"])
    )
    best_test_acc = 0.0
    best_val_loss = np.inf
    best_val_acc = 0
    pbar = trange(1, params["epochs"] + 1)

    writer = SummaryWriter(f"tensorboard/{model_save_name}")
    count = 0

    if not os.path.isdir(f"runs/{dataset_name}"):
        print(dataset_name)
        os.mkdir(f"runs/{dataset_name}")

    for epoch in pbar:
        # in every epoch we randomly choose a diameter between None, 1 - 7 and adjust the number of iterations

        loss = train(train_loader)

        # add training loss to tensorboard
        writer.add_scalar("stat/training-loss", loss, count)
        count += 1

        train_acc, _ = test(train_loader)
        val_acc, val_loss = test(val_loader)

        writer.add_scalar("stat/val-acc", val_acc, count)

        test_acc, _ = test(test_loader, iterations=iterations)

        writer.add_scalar("stat/test-acc-n-100", test_acc, count)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc



            torch.save(model, f"runs/{dataset_name}/{model_save_name}_{timestamp}.pt")
        print("\n")
        pbar.set_description(
            f"Epoch: {epoch:04d}, Loss: {loss:.3f} Train: {train_acc:.3f},"
            f" Val: {val_acc:.3f}, Test n=100: {test_acc:.3f}"
            f" Best Val|Test Acc.:  {best_val_acc:.3f} | {best_test_acc:.3f}",
        )

    print("---------- GNN (Acc, Loss) ----------")
    device = torch.device("cpu")
    model = torch.load(f"runs/{dataset_name}/{model_save_name}_{timestamp}.pt").to(device)

    # accuracies of best model
    train_acc = test(train_loader)[0]
    val_acc = test(val_loader)[0]

    print(f"Train: {train_acc}, Val: {val_acc}")

    for i,test_loader in enumerate(test_loaders):
        test_acc = test(test_loader, iterations= tests[i][1])[0]
        print(f"Extrapolation Softmax {test_acc:.3f} \n \n")

    print("---------- GNN-FSM HARDMAX ----------")

    sizes = [4, 10, 20, 50, 100]

    rows = []

    rows.append(['graphfsa', '1', params['seed'], val_acc, params['dataset_name']])

    i = 0
    for test_loader in test_loaders:
        test_1_acc = test(test_loader, iterations= tests[i][1], hardmax=True)[0]
        rows.append(['graphfsa', tests[i][0], params['seed'], test_1_acc, params['dataset_name']])

        i += 1
        print(f"n = {nodes} {test_1_acc}")

    df = pd.DataFrame(rows, columns=['model', 'graph_size', 'seed', 'test_acc', 'dataset'])

    dataset_name = params['dataset_name']
    seed = params['seed']
    df.to_csv(f'results/graphfsa_{dataset_name}_{seed}.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the State Machine GNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="Distance",
        help="Name of the dataset to run the experiment on",
    )
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--seed", default=42, type=int)  # add seed argument

    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--states_n", default=STATES_N, type=int)
    parser.add_argument("--patience", default=PATIENCE, type=int)

    args = parser.parse_args()
    STATES_N = args.states_n
    PATIENCE = args.patience

    params = {
        "learning_rate": args.learning_rate,#0.25,  #  0.0004,  # 0.01,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dataset_name": args.dataset,
        "iteration_offset": False,
        "iteration_offset_size": 1,
        "transition_memory": False,
        "weighted_loss": False,
        "additive_weight_factor": False,
        "multi_machine": False,
        "seed": args.seed,
    }

    dataset_name = args.dataset

    set_seed(args.seed)

    train_model(
        dataset_name,
        params,
    )
