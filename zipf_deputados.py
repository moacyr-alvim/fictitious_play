"""Zipf (ranking x votos) de candidatos a deputado, a partir dos dados do TSE.

Uso: python zipf_deputados.py votacao_candidato_munzona_2022.zip [saida_dir]
"""
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CARGOS = ["Deputado Federal", "Deputado Estadual", "Deputado Distrital"]
COLS = ["SG_UF", "DS_CARGO", "SQ_CANDIDATO", "NM_CANDIDATO", "QT_VOTOS_NOMINAIS"]


def agregar(zip_path):
    parts = []
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith("_BRASIL.csv"))
        with z.open(name) as f:
            for chunk in pd.read_csv(f, sep=";", encoding="latin1", usecols=COLS,
                                     chunksize=500_000):
                chunk = chunk[chunk["DS_CARGO"].isin(CARGOS)]
                parts.append(chunk.groupby(
                    ["SG_UF", "DS_CARGO", "SQ_CANDIDATO", "NM_CANDIDATO"],
                    as_index=False)["QT_VOTOS_NOMINAIS"].sum())
    df = pd.concat(parts).groupby(
        ["SG_UF", "DS_CARGO", "SQ_CANDIDATO", "NM_CANDIDATO"],
        as_index=False)["QT_VOTOS_NOMINAIS"].sum()
    return df.rename(columns={"QT_VOTOS_NOMINAIS": "votos"})


def ranquear(d):
    d = d.sort_values("votos", ascending=False).reset_index(drop=True)
    d["rank"] = d.index + 1
    return d


def main():
    zip_path = sys.argv[1]
    out = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
    out.mkdir(parents=True, exist_ok=True)
    df = agregar(zip_path)
    df.to_csv(out / "votos_por_candidato.csv", index=False)

    fed = df[df["DS_CARGO"] == "Deputado Federal"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (titulo, d) in zip(axs, [
        ("Brasil — Deputado Federal (2022)", ranquear(fed)),
        ("São Paulo — Deputado Federal (2022)", ranquear(fed[fed["SG_UF"] == "SP"])),
    ]):
        ax.loglog(d["rank"], d["votos"], ".", ms=3)
        ax.set(title=titulo, xlabel="Ranking (log)", ylabel="Votos (log)")
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "zipf_deputados.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 5))
    for uf, d in fed.groupby("SG_UF"):
        d = ranquear(d)
        ax.loglog(d["rank"], d["votos"], lw=.8, alpha=.7)
    ax.set(title="Deputado Federal 2022 — uma curva por UF",
           xlabel="Ranking (log)", ylabel="Votos (log)")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "zipf_por_uf.png", dpi=150)
    print(df.groupby(["DS_CARGO"]).size())


if __name__ == "__main__":
    main()
