from modules import *
from stable_baselines3 import PPO
from modules import GridFlexEnv
import pandas as pd
import os

## Nesse main é definido os diretórios, modo de operação, e dados necessários para rodar a simulação de fluxo de potência com BESS.
## No final é salvo os resultados em arquivos csv.

# Input and output paths    
path_xlsx = os.getcwd() + '/GridFlexPy/data/spreadsheets/'; os.makedirs(path_xlsx,exist_ok=True)
path_dss = os.getcwd() + '/GridFlexPy/data/dss_files/' ; os.makedirs(path_dss,exist_ok=True)
output_csv = os.getcwd() + '/GridFlexPy/data/output/csv/'; os.makedirs(output_csv,exist_ok=True)
output_img = os.getcwd() + '/GridFlexPy/data/output/img/'; os.makedirs(output_img,exist_ok=True)
path_generators = os.getcwd() + '/GridFlexPy/data/generators_profiles/'; os.makedirs(path_generators,exist_ok=True)

if __name__ == '__main__':

    config ={
        'name_spreadsheet': 'sheet_IEEE13Node.xlsx',
        'name_dss': 'CondominioDosIpes.dss',
        'kind': 'RLForecasting',  # Options: 'NoOperation', 'Simple', 'Smoothing', 'Forecasting', "RLForecasting"
        'bess_bus': 'bus_013',
        'seq_len': 3,  # Tamanho da janela de entrada
        'past_values': 3,  # Past values to start operation in 'Smoothing' or 'Simple' mode
        'in_feature': 3,  # Features in on mode Forcasting: Load, Losses and PV
        'n_future': 1,    # Number of future values to predict (1 for next timestep)
        'hidden_size': 64, # Hidden size of the LSTM
        'batch_size': 32, # Batch size for training
        'learning_rate': 1e-4, # Learning rate for the optimizer
        'num_layers': 3,
        'dropout': 0.2,
    }
    
    env = GridFlexEnv(config)
    model = PPO("MlpPolicy", env, verbose=1)        # ou PPO.load(...)

    model.learn(total_timesteps=288)             # <- treino de fato
    model.save("ppo_gridflex")

    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    results = env.episode_results()
    rewards = pd.DataFrame([r.as_dict for r in env.reward_trace])
    # Salva os resultados em arquivos CSV
    results['demand'].to_csv(os.path.join(output_csv, 'rl_forecasting_results.csv'), index=False)
    rewards.to_csv(os.path.join(output_csv, 'rl_forecasting_rewards.csv'), index=False)
    env.close()

    

    



    
    
