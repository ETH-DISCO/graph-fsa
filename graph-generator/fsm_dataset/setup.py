from setuptools import setup
  
setup(
    name='fsm_dataset',
    version='0.1',
    description='A GNN dataset generator based on deterministic FSM learning',
    author='Anonym',
    author_email='anonymous',
    packages=['fsm_dataset'],
    install_requires=[
        'networkx',
        'torch_geometric',
        'numpy',
        'torch',
        'graphviz',
    ],
)