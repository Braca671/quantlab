"""Genera notebooks/00_setup.ipynb."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# 00 — Setup e modello di riferimento

Esegui su Colab con GPU (Runtime → Cambia tipo di runtime → T4).
Alla fine avrai `checkpoints/smallcnn.pt` e l'accuratezza float32 di riferimento."""),
    code("""# Clona il tuo repo (sostituisci l'URL) e posizionati nella radice
import os, sys
if "google.colab" in sys.modules and not os.path.exists("quantlab"):
    !git clone https://github.com/<tuo-utente>/quantlab.git
    %cd quantlab
elif os.path.basename(os.getcwd()) == "notebooks":
    %cd ..
sys.path.insert(0, os.getcwd())          # da qui in poi si lavora dalla radice del repo"""),
    code("""!python scripts/train_cnn.py --epochs 20 --out checkpoints/smallcnn.pt"""),
    md("""Il checkpoint è perso quando la sessione Colab si chiude: salvalo su Google Drive."""),
    code("""# Facoltativo: copia il checkpoint su Drive
# from google.colab import drive; drive.mount("/content/drive")
# !mkdir -p /content/drive/MyDrive/quantlab && cp checkpoints/smallcnn.pt /content/drive/MyDrive/quantlab/"""),
    code("""import torch
from quantlab.models import SmallCNN
from quantlab.data import cifar10_loaders
from quantlab.evaluate import accuracy, model_size_bytes

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SmallCNN(); model.load_state_dict(torch.load("checkpoints/smallcnn.pt", map_location=device))
_, test_loader = cifar10_loaders()
print(f"accuratezza float32: {accuracy(model, test_loader, device):.4f}")
for b in [None, 8, 4, 3, 2]:
    print(f"pesi a {b or 32:>2} bit: {model_size_bytes(model, b) / 1024:7.1f} KiB")"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "00_setup.ipynb")
