# multitask GP

### 1. train the model
```bash
cd <project_root>/imputation/models/multitaskGP
python train.py
```

### 2. analyse the trained model
1. modify the value of variables in analysis.py
```python
# (optional) if you want to visualise the result, set the following variable to True
VISUALISE_FOR_DEBUG = True

# (mandatory) set the path to the saved model
best_model_path = "./model_weights/<path to the model>"

# (mandatory) set the target time points for query
target_ts = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5])
```

2. run the code to know how to do imputation via the trained model
```bash
cd <project_root>/imputation/models/multitaskGP
python analysis.py
```
