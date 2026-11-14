# Kaggle dataset example model and analysis

## Files:
    data_understand.py : Analysis of the kaggle dataset
    energy.ipynb : Model on the data (Multi Linear regression)

# Dataset

The dataset for the project is: 
**<a href="https://www.kaggle.com/datasets/elikplim/eergy-efficiency-dataset">Kaggle dataset</a>**

# **Functions in** _"data_understand.py"_
    load_data(path) : Loads data without filter applied

    filter1(data)   : Transforms string data to `np.array`

    get_data(path, batch_size) : Gets batch_size of examples with filter applied

# Data
### data.shape: (784, 10)
## Inputs: (784, 8)
    1. Relative Compactness
    2. Surface Area
    3. Wall Area
    4. Roof Area
    5. Overall Height
    6. Orientation
    7. Glazing Area"
    8. Glazing Area Distribution
## Outputs: (784, 2)
    1. Heating Load
    2. Cooling Load


# About Model
## Model Architecture
    Sequential(
        Linear(8, 512), ReLU,
        Linear(512, 512), ReLU,
        Linear(512, 2)
    )

### Data Split : 
**90%** -> Train(691, 10)    
**10%** -> Test(93, 10)

I have normalized the data columns wherever `data>1`

## Optimizer: AdamW(..., lr=3e-4)
## Loss: L1Loss()

# Best Val:

    Training Loss: 0.25406
    Eval loss: 0.29400

# [REMARK]
No AI at any point of time has been used for anything except for one time only to realize that i forgot 
```python
optimizer.zero_grad(set_to_none=True)
```
and that my implementation of **MAE** was a bit wrong idk why, and the way i can use **MAE** in torch is by
```python
loss_fn = nn.L1Loss()
```
And,
I use `T4 colab GPU` for `100_000 EPOCHS` run
