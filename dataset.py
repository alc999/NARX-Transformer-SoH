from torch.utils.data import Dataset

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