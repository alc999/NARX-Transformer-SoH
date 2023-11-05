import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_capacity_sequence(model, battery_dict, num_cycles):
    for name in battery_dict:
        soh_labels = []
        soh_preds = []
        data, caps = battery_dict[name]['data'], battery_dict[name]['cap']
        for i in range(len(data)-num_cycles):
            input_stream = data[i:i+num_cycles]
            soh_labels.append(caps[i+num_cycles-1])
            input_stream = torch.tensor(input_stream).unsqueeze(0).float()
            cap_stream = torch.tensor(caps[i:i+num_cycles-1]).unsqueeze(0).float()
            pred = model(input_stream, cap_stream)
            soh_preds.append(pred.detach().numpy())
        soh_labels = np.array(soh_labels).ravel()
        soh_preds = np.array(soh_preds).ravel()
        plt.plot(soh_labels)
        plt.plot(soh_preds)
        plt.show()
        break

def plot_predicted_capacity(model, train_dataset, test_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    battery_types = train_dataset.battery_data[:,0]

    for battery_type in np.unique(battery_types):
        train_data = train_dataset.battery_data[train_dataset.battery_data[:,0] == battery_type]
        test_data = test_dataset.battery_data[test_dataset.battery_data[:,0] == battery_type]
        
        train_caps = [c[-1] for c in train_data[:,-1]]
        test_caps = [c[-1] for c in test_data[:,-1]]
        pred_caps = [test_data[:,-1][0][-1], test_data[:,-1][1][-1]]
        for t_data in test_data:
            _, inputs, _ = t_data
            inputs = torch.tensor(inputs).float().unsqueeze(0).to(device)
            outputs = torch.tensor(pred_caps[-2:]).float().unsqueeze(0).to(device)
            # outputs = torch.tensor(outputs).float().unsqueeze(0).to(device)
            pred = model(inputs, outputs)
            pred_caps.append(pred.item())

        plt.plot(train_caps+[test_caps[0]], 'blue')
        plt.plot(range(len(train_caps),len(train_caps)+len(test_caps)), test_caps, 'green')
        plt.plot(range(len(train_caps),len(train_caps)+len(pred_caps)), pred_caps, 'red')
        plt.grid('on')
        plt.legend(['Real','Label','Predicted'])
        plt.xlabel('Cycle Number')
        plt.ylabel('SoH (%)')
        plt.show()