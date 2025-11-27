from modules import *
from stable_baselines3 import PPO
from modules import GridFlexEnv, TrainLoggingCallback
import pandas as pd
import os
import torch as th

## Nesse main é definido os diretórios, modo de operação, e dados necessários para rodar a simulação de fluxo de potência com BESS.
## No final é salvo os resultados em arquivos csv.

# Input and output paths
path_xlsx = os.getcwd() + "/data/spreadsheets/"
os.makedirs(path_xlsx, exist_ok=True)
path_dss = os.getcwd() + "/data/dss_files/"
os.makedirs(path_dss, exist_ok=True)
output_csv = os.getcwd() + "/data/output/csv/"
os.makedirs(output_csv, exist_ok=True)
output_img = os.getcwd() + "/data/output/img/"
os.makedirs(output_img, exist_ok=True)
path_generators = os.getcwd() + "/data/generators_profiles/"
os.makedirs(path_generators, exist_ok=True)
dir_model = os.getcwd() + "/data/output/csv/models/"
os.makedirs(dir_model, exist_ok=True)


if __name__ == "__main__":
    base_config = {
        "name_spreadsheet": "sheet_IEEE13Node.xlsx",
        "name_dss": "CondominioDosIpes.dss",
        "kind": "RLForecasting",  # Options: 'NoOperation', 'Simple', 'Smoothing', 'Forecasting', "RLForecasting"
        "bess_bus": "bus_013",
        "seq_len": 3,  # Tamanho da janela de entrada
        "past_values": 3,  # Past values to start operation in 'Smoothing' or 'Simple' mode
    }
    ######## ----------------------- Parâmetros ------------------- ################
    # Dados completos (os mesmos da planilha)
    reward_weights = {"delta_sigma": 700.0, "delta_norm": 1000.0, "soc": 500.0}
    policy_kwargs = dict(
    activation_fn=th.nn.Tanh,   # opcional, padrão já é ReLU
    net_arch=[dict(
        pi=[6, 6],          # camadas da policy (π)
        vf=[6, 6]           # camadas da value function (V)
    )],
)
    full_start = pd.Timestamp("2012-07-07 06:00")
    full_end = pd.Timestamp("2014-02-28 00:00")
    dt_minutes = 5
    split_test = 0.7  

    full_range = pd.date_range(full_start, full_end, freq=f"{dt_minutes}T")
    n_total = len(full_range)

    n_train = int(n_total * split_test)
    split_timestamp = full_range[n_train]  # fim do conjunto de treino
    test_start = split_timestamp + pd.Timedelta(minutes=dt_minutes)
    ######## ----------------------- Configuração dos ambientes ------------------- ################
    # Ambiente de treino
    train_config = base_config.copy()
    train_config["start_date"] = full_start
    train_config["end_date"] = split_timestamp

    env_train = GridFlexEnv(train_config, reward_weights=reward_weights)
    N_timestamps = len(env_train.time_range)
    warmup_steps = env_train.warmup_steps
    steps_por_episodio = N_timestamps - warmup_steps 

    # # Ambiente de teste
    test_config = base_config.copy()
    test_config["start_date"] = test_start
    test_config["end_date"] = full_end
    env_test = GridFlexEnv(test_config,reward_weights=reward_weights)


    # full_config = base_config.copy()
    # full_config["start_date"] = full_start
    # full_config["end_date"]   = full_end

    # env_full = GridFlexEnv(full_config, reward_weights=reward_weights)

    ######## ----------------------- Treinamento ------------------- ################
    model = PPO("MlpPolicy", env_train, verbose=1, n_steps=steps_por_episodio)
    train_callback = TrainLoggingCallback(env_train,save_dir= output_csv, render_every_n=1,verbose=1)  # por exemplo, print a cada 1 steps
    print(
        f"Iniciando o treinamento do agente RL em {full_start} até {split_timestamp}..."
    )
    print("N_timestamps       =", N_timestamps)
    print("warmup_steps       =", warmup_steps)
    print("steps_por_episodio =", steps_por_episodio)

    model.learn(total_timesteps=steps_por_episodio, callback=train_callback)  #
    # Salvando o modelo e os índices
    print("Salvando o modelo treinado...")
    model.save(os.path.join(dir_model, "ppo_gridflex"))

    ######## ----------------------- Teste ------------------- ################

    # Avaliação do agente treinado no ambiente de teste
    # print(f"Avaliando o agente RL treinado em {full_start} até {full_end}...")

    # model = PPO.load(os.path.join(dir_model, "ppo_gridflex"), env=env_full)

    # obs, info = env_full.reset()
    # done = False

    # while not done:
    #     action, _ = model.predict(obs, deterministic=True)
    #     print(f"Potência da bateria em {env_full.time_range[env_full.current_idx]}: P(kW) = {action}")
    #     obs, reward, terminated, truncated, info = env_full.step(action)
    #     done = terminated or truncated

    # results = env_full.episode_results()
    # rewards = pd.DataFrame([r.as_dict for r in env_full.reward_trace])

    # # Salvando resultados em arquivos CSV
    # print("Salvando os resultados em arquivos CSV...")
    # indices_df = env_full.indices_results()
    # indices_df.to_csv(os.path.join(dir_model, "indices_test.csv"), index=False)
    # results["demand"].to_csv(
    #     os.path.join(output_csv + "demand/", "rl_forecasting_results_test_PPO.csv"),
    #     index=False,
    # )
    # results["bess"].to_csv(
    #     os.path.join(output_csv + "bess/", "rl_forecasting_bess_power_test_PPO.csv"),
    #     index=False,
    # )
    # rewards.to_csv(
    #     os.path.join(dir_model, "rl_forecasting_rewards_test_PPO.csv"), index=False
    # )

    # env_full.close()

    print(f"Avaliando o agente RL treinado em {test_start} até {full_end}...")

    model = PPO.load(os.path.join(dir_model, "ppo_gridflex"), env=env_test)

    obs, info = env_test.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        print(f"Potência da bateria em {env_test.time_range[env_test.current_idx]}: P(kW) = {action}")
        obs, reward, terminated, truncated, info = env_test.step(action)
        done = terminated or truncated

    results = env_test.episode_results()
    rewards = pd.DataFrame([r.as_dict for r in env_test.reward_trace])

    # Salvando resultados em arquivos CSV
    print("Salvando os resultados em arquivos CSV...")
    indices_df = env_test.indices_results()
    indices_df.to_csv(os.path.join(dir_model, "indices_test.csv"), index=False)
    results["demand"].to_csv(
        os.path.join(output_csv + "demand/", "rl_forecasting_results_test_PPO.csv"),
        index=False,
    )
    results["bess"].to_csv(
        os.path.join(output_csv + "bess/", "rl_forecasting_bess_power_test_PPO.csv"),
        index=False,
    )
    rewards.to_csv(
        os.path.join(dir_model, "rl_forecasting_rewards_test_PPO.csv"), index=False
    )

    env_test.close()
