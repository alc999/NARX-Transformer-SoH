import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

def load_NASA(folder, num_cycles, split_ratio, scale_data=True):
    battery_dict = load_raw_data(folder=folder, scale_data=True)
    battery_data = []
    for battery_type in battery_dict:
        for i in range(len(battery_dict[battery_type]['data'])-num_cycles+1):
            sequences = battery_dict[battery_type]['data'][i:i+num_cycles]
            capacities = battery_dict[battery_type]['cap'][i:i+num_cycles]
            battery_data.append([battery_type,sequences,capacities])

    train_data, test_data = sequential_train_test_split(battery_data, split_ratio)
    train_dataset, test_dataset = BatteryDataset(train_data), BatteryDataset(test_data)
    return train_dataset, test_dataset

def load_raw_data(folder, scale_data=True):
    charge_data = pd.read_csv(folder+'/charge_data.csv')
    data_x = np.load(folder+'/data_x.npy')
    data_y = np.load(folder+'/data_y.npy')

    if scale_data:
        sc_voltage = MinMaxScaler(feature_range=(-1,1))
        sc_current = MinMaxScaler(feature_range=(-1,1))
        sc_temp    = MinMaxScaler(feature_range=(-1,1))

        data_x[:,:,0] = sc_voltage.fit_transform(data_x[:,:,0])
        data_x[:,:,1] = sc_current.fit_transform(data_x[:,:,1])
        data_x[:,:,2] = sc_temp.fit_transform(data_x[:,:,2])

    battery_dict = {}
    for name in np.unique(charge_data['Battery_id']):
        battery_dict[name] = {}
        battery_dict[name]['data'] = data_x[charge_data[charge_data['Battery_id']==name].index]
        battery_dict[name]['cap']  = data_y[charge_data[charge_data['Battery_id']==name].index]

        idx,_,_ = np.where(np.isnan(battery_dict[name]['data']))
        battery_dict[name]['data'] = np.delete(battery_dict[name]['data'], idx, axis=0)
        battery_dict[name]['cap'] = np.delete(battery_dict[name]['cap'], idx, axis=0)
    return battery_dict

def sequential_train_test_split(battery_data, split_ratio):
    battery_data = np.array(battery_data, dtype=object)
    battery_types = battery_data[:,0]

    train_data = []
    test_data = []
    for battery_type in np.unique(battery_types):
        one_type_data = battery_data[battery_data[:,0] == battery_type]
        one_type_train, one_type_test = train_test_split(one_type_data, test_size=split_ratio, shuffle=False)
        train_data.append(one_type_train)
        test_data.append(one_type_test)

    train_data = np.vstack(train_data)
    test_data = np.vstack(test_data)
    return train_data, test_data

class BatteryDataset(Dataset):
    def __init__(self, battery_data):
        self.battery_data = battery_data

    def __len__(self):
        return len(self.battery_data)

    def __getitem__(self, index):
        name, data, caps = self.battery_data[index]
        return data, caps

##################################### dev #####################################

class old_BatteryDataset(Dataset):
    def __init__(self, battery_dict, num_cycles):
        self.battery_data = []
        for battery_type in battery_dict:
            for i in range(len(battery_dict[battery_type]['data'])-num_cycles+1):
                sequences = battery_dict[battery_type]['data'][i:i+num_cycles]
                capacities = battery_dict[battery_type]['cap'][i:i+num_cycles]
                self.battery_data.append([battery_type,sequences,capacities])

    def __len__(self):
        return len(self.battery_data)

    def __getitem__(self, index):
        name, data, caps = self.battery_data[index]
        return data, caps
