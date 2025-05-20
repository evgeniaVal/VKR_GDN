import torch

dataset = 'SWaT'                  
checkpoint_path = f"data/swat/best.pt"
list_path = "data/swat/list.txt"     
output_file = "emb.txt"           
 
state = torch.load(checkpoint_path, map_location='cpu')
state_dict = state.get('state_dict', state)
 
embed_key = next(
    (k for k in state_dict if 'emb' in k.lower()), 
    None
)
if embed_key is None:
    raise RuntimeError("Не найден слой embedding в чекпоинте")
 
embs = state_dict[embed_key].cpu().numpy()  # shape: (num_sensors, embed_dim)
 
with open(list_path, 'r') as f:
    sensors = [line.strip() for line in f if line.strip()]
 
if len(sensors) != embs.shape[0]:
    raise RuntimeError("Число сенсоров в list.txt не совпадает с числом эмбеддингов")
 
with open(output_file, 'w') as fout:
    for name, vec in zip(sensors, embs):
        vec_str = " ".join(f"{v:.6f}" for v in vec)
        fout.write(f"{name} {vec_str}\n")
 
print(f"Эмбеддинги сохранены в {output_file}")