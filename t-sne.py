import torch
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

dataset = 'swat'
checkpoint = f"data/swat/best.pt"
list_txt   = "data/swat/list.txt"        
out_png    = f"{dataset}_tsne_labels_colored.png"
 
state = torch.load(checkpoint, map_location='cpu')
sd = state.get('state_dict', state)
 
emb_key = next((k for k in sd if 'emb' in k.lower()), None)
if emb_key is None:
    raise RuntimeError("Не найден слой embedding")
embs = sd[emb_key].cpu().numpy().astype(np.float32)
 
with open(list_txt) as f:
    sensors = [l.strip() for l in f if l.strip()]
if len(sensors) != embs.shape[0]:
    raise RuntimeError("Число сенсоров и эмбеддингов не совпадает")
 
classes = []
for name in sensors:
    m = re.match(r'^([A-Za-z]+)', name)
    classes.append(m.group(1) if m else name)
uniq = sorted(set(classes))
cls2idx = {c: i for i, c in enumerate(uniq)}
labels = np.array([cls2idx[c] for c in classes])
 
embs = StandardScaler().fit_transform(embs)
 
perp = min(10, len(embs)//3)
tsne = TSNE(
    n_components=2,
    perplexity=perp,
    init='pca',
    learning_rate=200.0,
    n_iter=1000,
    random_state=42
)
proj = tsne.fit_transform(embs)
 
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    proj[:, 0], proj[:, 1],
    c=labels, cmap='tab10',
    s=40, edgecolors='w', linewidths=0.5
)

handles, _ = scatter.legend_elements()
plt.legend(handles, uniq, title="Sensor Class", bbox_to_anchor=(1.05, 1), loc='upper left')
 
y_offset = (proj[:,1].max() - proj[:,1].min()) * 0.01  # 1% от y-диапазона
for i, name in enumerate(sensors):
    plt.text(
        proj[i, 0], proj[i, 1] + y_offset,
        name, fontsize=6, va='bottom', ha='center'
    )
 
plt.title(f"t-SNE of {dataset} Sensor Embeddings")
plt.tight_layout()
plt.savefig(out_png, dpi=300)
plt.show()
 
print(f"Сохранено: {out_png}")