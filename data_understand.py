import numpy as np
def load_data(path, batch_size):
    raw_data = open(path, 'r').readlines()[1:batch_size]   # specific to the data i have
    visited = []
    for i in raw_data:
        if i not in visited:
            visited.append(i)        ### No duplication in data
    return visited

def filter1(data):
    # filter the data
    return np.array([i[:-1].split(',') for i in data], dtype=np.float32)

def get_data(path, batch_size: int = 32) -> np.ndarray:
    """
    path: path to dataset
    batch_size: how many examples you want

    return: numpy array of shape -> (batch_size, 10)
    """
    data = load_data(path, batch_size)
    return filter1(data)

if __name__ == "__main__":

    raw_data = load_data("/home/aditya0oo0/.cache/kagglehub/datasets/elikplim/eergy-efficiency-dataset/versions/1/ENB2012_data.csv")
    data = filter1(raw_data)

    # data dims in dataset
    specs = ["Relative Compactness", "Surface Area", "Wall Area", "Roof Area", "Overall Height", "Orientation", "Glazing Area", "Glazing Area Distribution", "Heating Load", "Cooling Load"]
    print(f"First row of the dataset: {data[0]}\n\n")
    print("Inputs to potential modelling-------------------")
    for s, i in zip(specs[:-2], data[0, :-2]):
        print(f"{s}: {i:.4f}")
    print("\n")
    
    print("Output to above inputs--------------------------")
    for s, i in zip(specs[-2:], data[0, -2:]):
        print(f"{s}: {i:.4f}")
    print("\n")
    
    print("Dimension of whole dataset: ", data.shape)
    
    
