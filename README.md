# Simplex-Flows

```py
python -m src.train experiment=toy_dfm_sfm_cnn
```
## Promoter and Enhancer DNA Experiment

To run the dirichlet flow model on the promoter and enhancer dataset:

```py
python -m src.train experiment=promoter_dfm logger=wandb
```

## RetroBridge: Products and Reactants Experiment

Test whether the data has been loaded correctly:

```py
python -m src.data.retrobridge_datamodule
```

Test whether the model has been correctly coded up:

```py
python -m src.models.retrobridge_module
```


To run RetroBridge on this dataset:

TODO:

```py
python -m src.train experiment=retrobridge_retrosyn logger=wandb
```



## Text8

```py
python -m src.train experiment=text8_sfm_bmlp logger=wandb
```

## QM9

Download trained models

```sh
wget -r -np -nH --cut-dirs=2 --reject 'index.html*' https://bits.csb.pitt.edu/files/FlowMol/trained_models/
```

Download the QM9 dataset

```sh
mkdir data/qm9_raw
cd data/qm9_raw
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip
wget -O uncharacterized.txt https://ndownloader.figshare.com/files/3195404
unzip qm9.zip
```

Process the dataset:
```py
python process_qm9.py --config=trained_models/qm9_gaussian/config.yaml
```

To test the DataModule:

```py
python -m src.data.qm9_datamodule
```