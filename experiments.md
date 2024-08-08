#File to reproduce all experiments




## State Automatons

### GOL
GraphFSA: `python train-gol.py --batch_size=1 --epochs=5 --learning_rate=0.5 --patience=2  --seed=42 --states_n=2 --toroidal=True --grid_size=4 --steps=1 --model=graphfsa`
RecGNN: `python train-gol.py --batch_size=32 --epochs=5 --learning_rate=0.001 --patience=2 --clip=True --seed=42 --states_n=2 --toroidal=True --grid_size=4 --steps=1 --model=recgnn`
GNCA:: `python train-gol.py --batch_size=32 --epochs=5 --learning_rate=0.001 --patience=2 --clip=True --seed=42 --states_n=2 --toroidal=True --grid_size=4 --steps=1 --model=gnca`

GraphChef: `python train-gol.py --batch_size=64 --epochs=250 --learning_rate=0.0006 --patience=20 --clip=True --seed=42 --states_n=2 --toroidal=True --grid_size=4 --steps=1 --model=graphchef` (0, loss but not during inf)

### 1D
GraphFSA: `python train-1dca.py --batch_size=1 --epochs=5 --learning_rate=0.5 --patience=2  --seed=42 --states_n=2 --grid_size=4 --steps=1 --model=graphfsa`
RecGNN: `python train-1dca.py --batch_size=32 --epochs=50 --learning_rate=0.001 --patience=2 --clip=True --seed=42 --states_n=2  --grid_size=4 --steps=1 --model=recgnn` (trains ok, doesnt fit perfect, i think it cant bc of aggregation)
GNCA: `python train-1dca.py --batch_size=32 --epochs=50 --learning_rate=0.001 --patience=2 --clip=True --seed=42 --states_n=2 --grid_size=4 --steps=1 --model=gnca` (also reaches ~85%)
GraphChef: ``

### Wireworld
GraphFSA: `python train-wireworld.py --batch_size=1 --epochs=5 --learning_rate=0.5 --patience=2  --seed=42 --states_n=4 --grid_size=4 --steps=1 --model=graphfsa`
RecGNN: `python train-wireworld.py --batch_size=32 --epochs=50 --learning_rate=0.001 --patience=2 --clip=True --seed=42 --states_n=4  --grid_size=4 --steps=1 --model=recgnn` 
GNCA: `python train-wireworld.py --batch_size=32 --epochs=50 --learning_rate=0.001 --patience=5 --clip=True --seed=42 --states_n=4 --grid_size=4 --steps=1 --model=gnca` (tops out at ~95 percent)

GraphChef: ``

### GRAB
GraphFSA: `python train_graphfsa.py --batch_size=1 --epochs=20 --learning_rate=0.25 --patience=2  --seed=42 --states=4`
GraphFSA: `python train_graphfsa.py --batch_size=1 --epochs=20 --learning_rate=0.25 --patience=2  --seed=42 --states=6`
RecGNN: `python train_baseline.py --batch_size=32 --epochs=100 --learning_rate=0.001 --patience=10 --clip=True --seed=42 --model=recgnn` 
GNCA: `python train_baseline.py --batch_size=32 --epochs=100 --learning_rate=0.001 --patience=10 --clip=True --seed=42 --model=gnca --steps=5` (doesn't work)

GraphChef: ``

### Algorithms
GraphFSA: `python train.py --batch_size=1 --epochs=10 --learning_rate=0.25 --patience=2  --seed=42 --dataset=PrefixSum`
RecGNN: `python train_baselines.py --batch_size=32 --epochs=20 --learning_rate=0.001 --patience=5 --seed=42  --dataset=Distance --model=recgnn` 
GNCA: `python train_baselines.py --batch_size=32 --epochs=20 --learning_rate=0.001 --patience=5 --seed=42  --dataset=Distance --model=gnca` 

GraphChef: ``
