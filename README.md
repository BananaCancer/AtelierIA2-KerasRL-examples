# Deep Reinforcement Learning for Keras

## Informations
Ce dossier s'inspire des exemples proposés par le repo github <a href="https://github.com/keras-rl/keras-rl">kerasRL</a>.

Dans le dossier src, sont présents les 3 fichiers utilisés pour entraîner les jeux Taxi, FrozenLake et CliffWalking. Dans le dossier logs sont présents les logs des entraînements.

## Dépendances
Pour pouvoir le faire tourner sur windows, j'ai dû effectuer les commandes suivantes
```
conda create --name kerasrl
conda activate kerasrl
conda install python==3.7.2
pip install keras-rl keras gym[all] Pillow h5py
pip install tensorflow
pip install keras-rl2==1.0.4
pip install gym==0.25.2
pip install gym[atari,accept-rom-license]==0.21.0
pip install importlib-metadata==4.13.0
```

Si des erreurs liées à la fonction `image_dim_ordering` apparraissent (comme pour le fichier `dqn_atari.py`), il faut faire les changements suivants : 
```py
# Deprecated :
#K.image_dim_ordering() == 'tf'
# Replace with K.image_data_format() == 'channels_last'
#K.image_dim_ordering() == 'th'
# Replace with K.image_data_format() == 'channels_first'
```
