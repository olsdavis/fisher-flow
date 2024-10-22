# Fisher Flow Matching
All our dependencies are listed in `environment.yaml`, for Conda, and `requirements.txt`, for `pip`. Please also separately install `DGL`:
```bash
pip install -r requirements.txt
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
```
Our code contains parts of [FlowMol](https://github.com/Dunni3/FlowMol/tree/main) by Dunn and Koes [1] (most of QM9 experiments), [Riemannian-FM](https://github.com/facebookresearch/riemannian-fm) by Chen, et al. [2], and, for the baselines, [DFM](https://github.com/HannesStark/dirichlet-flow-matching/tree/main) by Stark, et al [3].

## Toy Experiment
For the DFM toy experiment, the following command allows us to run our code:
```bash
python -m src.train experiment=toy_dfm_bmlp data.dim=100 trainer=gpu trainer.max_epochs=500
```
Of course, the dimension argument is varied, and the configuration files allow for changing manifolds (`"simplex"`, or `"sphere"`) and turn OT on/off (`"exact"` or `"None"`).

## Promoter and Enhancer DNA Experiment
To download the datasets, it suffices to follow the steps of [Stark, et al](https://github.com/HannesStark/dirichlet-flow-matching/). For evaluating the FBD, it also needed to download their weights from their `workdir.zip`. To run the promoter dataset experiment, the following command can be used:

```bash
python -m src.train experiment=promoter_sfm_promdfm trainer.max_epochs=200 trainer=gpu data.batch_size=128
```

As for the enhancer MEL2 experiment, the following command is available:

```bash
python -m src.train experiment=enhancer_mel_sfm_cnn trainer.max_epochs=800 trainer=gpu
```

and for the FlyBrain DNA one:
```bash
python -m src.train experiment=enhancer_fly_sfm_cnn trainer.max_epochs=800 trainer=gpu
```

## QM9 experiment
To install the QM9 dataset, we have included `process_qm9.py` from FlowMol, so it suffices to follow the steps indicated in their [README](https://github.com/Dunni3/FlowMol/tree/main).

```bash
python -m src.train experiment=qm_clean_sfm trainer=gpu
```

## References
- [1]: [Dunn and Koes: Mixed Continuous and Categorical Flow Matching for 3D De Novo Molecule Generation](https://arxiv.org/abs/2404.19739).
- [2]: [Chen, et al.: Flow Matching on General Geometries](https://arxiv.org/pdf/2302.03660).
- [3]: [Stark, et al.: Dirichlet Flow Matching with Applications to DNA Sequence Design](https://arxiv.org/abs/2402.05841).
