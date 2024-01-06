import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
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

def plot_predicted_capacity(model, train_dataset, test_dataset, num_cycles):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    battery_types = train_dataset.battery_data[:,0]

    evaluations = []
    for battery_type in np.unique(battery_types):
        train_data = train_dataset.battery_data[train_dataset.battery_data[:,0] == battery_type]
        test_data = test_dataset.battery_data[test_dataset.battery_data[:,0] == battery_type]
        try:
            train_caps = [c[-1] for c in train_data[:,-1]]
            test_caps = [c[-1] for c in test_data[:,-1]]
            pred_caps = [train_data[:,-1][i][-1] for i in np.arange(-num_cycles+1,0)]
            
            for cycle in range(len(test_data)):
                _, inputs, _ = test_data[cycle]
                inputs = torch.tensor(inputs[:num_cycles]).float().unsqueeze(0).to(device)
                outputs = torch.tensor(pred_caps[-num_cycles+1:]).float().unsqueeze(0).to(device)
                pred = model(inputs, outputs)
                pred_caps.append(pred.item())
            
            evaluations.append([battery_type, np.array(test_caps), np.array(pred_caps[num_cycles-1:])])

            plt.figure(figsize=(8,5))
            plt.plot(train_caps+[test_caps[0]])
            plt.plot(range(len(train_caps),len(train_caps)+len(test_caps)), test_caps, c='olive')
            plt.plot(range(len(train_caps),len(train_caps)+len(pred_caps)-num_cycles+1), pred_caps[num_cycles-1:], '--', c='red')
            plt.grid('on')
            plt.title(battery_type, fontsize=20)
            plt.legend(['_','Real data','Prediction'], loc='upper right')
            plt.xlabel('Cycle', fontsize=15)
            plt.ylabel('Battery SoH (%)', fontsize=15)
            plt.show()
        except:
            pass

    return evaluations


def plot_error_histogram(errors):
    plt.figure(figsize=(8,5))
    plt.xlim([-0.1,0.1])
    counts, bins, _ = plt.hist(errors, bins=10, density=False, alpha=0.7, label='Error Distribution')

    # Fit a Gaussian distribution to the data
    mu, std = norm.fit(errors)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std) * np.sum(counts) / 200  
    plt.plot(x, p, 'k', linewidth=2, label=f'Gaussian Fit\n$\mu={mu:.2f}$, $\sigma={std:.2f}$')

    plt.xlabel('Error', fontsize=15)
    plt.ylabel('Counts', fontsize=15)
    # plt.title('Distribution of Errors with Gaussian Fit')
    plt.legend(loc='upper right')
    plt.show()
