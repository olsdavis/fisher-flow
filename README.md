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