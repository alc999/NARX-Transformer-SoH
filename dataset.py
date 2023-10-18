from torch.utils.data import Dataset, DataLoader

class BatteryDataset(Dataset):
    def __init__(self, battery_data):
        self.battery_data = battery_data

    def __len__(self):
        return len(self.battery_data)

    def __getitem__(self, index):
        name, data, cap = self.battery_data[index]

        if index > 0:
            old_name, _, old_cap = self.battery_data[index-1]
            if old_name!=name:
                old_cap=1
        else:
            old_cap = 1

        return data, old_cap, cap