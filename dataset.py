import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

def load_NASA(folder, scale_data=True):
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
    
class BatteryDataset(Dataset):
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
