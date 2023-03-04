import numpy as np
np.set_printoptions(suppress=True)
def split_data(data,split_size = [.8,.2]):
    lenght_data = data.shape[0]
    size_train = split_size[0]
    train_size = round(lenght_data*size_train)
    test_size = lenght_data - train_size
    train_idx = np.random.choice(lenght_data, size=train_size, replace=False)
    p_val_bs = np.ones(lenght_data)
    p_val_bs[train_idx] = 0
    p_val_bs = p_val_bs*1/test_size
    test_idx = np.random.choice(lenght_data, size=test_size, replace=False, p = p_val_bs)
    train_data,test_data = data[train_idx],data[test_idx]
    train_data,train_labels = train_data[...,:-1],train_data[...,-1]
    test_data,test_labels = test_data[...,:-1],test_data[...,-1]
    return train_data,train_labels,test_data,test_labels
