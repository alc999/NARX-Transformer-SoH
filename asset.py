import matplotlib.pyplot as plt

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
