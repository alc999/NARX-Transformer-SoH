import matplotlib.pyplot as plt

def plot_capacity_sequence(model, battery_dict, num_cycles):
    for name in battery_dict:
        soh_labels = []
        soh_preds = []
        data, caps = battery_dict[name]['data'], battery_dict[name]['cap']
        for i in range(len(data)-num_cycles):
            input_stream = data[i:i+num_cycles]
            soh_labels.append(caps[i+num_cycles-1])
            # pred = model(input_stream, caps[:-1])
            # soh_preds.append(pred)
        plt.plot(soh_labels)
        plt.show()
        break
