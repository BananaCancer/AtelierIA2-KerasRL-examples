import json
import matplotlib.pyplot as plt
import numpy as np

def plot_info(data, label, title, xlim=None, global_mean_start = 5, conv_mean_start=13000):
    if xlim == None:
        xlim=len(data)
    global_mean = np.mean(data[global_mean_start:xlim])
    end_mean = np.mean(data[conv_mean_start:xlim])
    plt.figure(figsize=(10,6))
    plt.plot(data, label=label)
    plt.hlines(global_mean, label=f"Moyenne globale : {global_mean}", xmin=0, 
            xmax=len(data), colors="red")
    

    plt.xlabel("Episode")
    plt.ylabel(label)
    plt.xlim([0, xlim])
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    with open('dqn_FrozenLake-v1_log.json', 'r') as file:
        data = json.load(file)
    plot_info(data["loss"], label="Loss", xlim=15000,
              title="Evolution de l'erreur du réseau de neurones du DQN pour le jeu FrozenLake.", conv_mean_start=450, global_mean_start=33)
    plot_info(data["duration"], label="Duration (s)", xlim=15000,
              title="Evolution du temps d'exécution des épisodes pour le jeu FrozenLake.", conv_mean_start=450)
    plot_info(data["nb_episode_steps"], label="Step count", xlim=15000,
              title="Evolution du nombre de step par épisode pour le jeu FrozenLake.", conv_mean_start=450)
    #plot_time(["mae"])