# Simplex-Flows

TODO:

## Promoter and Enhancer DNA Experiment

TODO:

```py
python xxx.py --arg1 --arg2
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
HYDRA_FULL_ERROR=1 python -m src.train experiment=retrobridge_retrosyn logger=wandb
```



## Text8

```py
python -m src.train experiment=text8_sfm_bmlp logger=wandb
```