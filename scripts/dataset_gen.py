"""Automatic dataset generator for SSLE-1 (.sds2 format).

Produces a balanced, themed corpus of Portuguese content titles across many
domains using compositional templates. Designed to create a robust training
set (thousands of samples) for title/headline generation.

Usage:
    python scripts/dataset_gen.py --count 6000 --output data/dataset.sds2
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List

# ---------------------------------------------------------------------- #
# Themes and their domain vocabulary.
# ---------------------------------------------------------------------- #
THEMES: Dict[str, Dict[str, List[str]]] = {
    "FORTNITE": {
        "context": ["MIRA", "CONSTRUCAO", "RANKED", "DICAS", "SKINS", "VITORIA", "BATTLE ROYALE"],
        "objeto": ["MIRA", "CONSTRUCAO", "MOVIMENTACAO", "EDICAO", "ESTRATEGIA", "GAMEPLAY"],
        "plataforma": ["FORTNITE", "PC", "CONSOLE", "MOBILE"],
        "acao": ["MELHORAR", "DOMINAR", "TREINAR", "EVOLUIR", "APRIMORAR"],
    },
    "MINECRAFT": {
        "context": ["SURVIVAL", "FARM", "REDSTONE", "BUILD", "MODS", "SERVIDOR"],
        "objeto": ["CASA", "FARM", "FAZENDA", "PORTAL", "MINERACAO", "REDSTONE"],
        "plataforma": ["MINECRAFT", "JAVA", "BEDROCK", "SERVIDOR"],
        "acao": ["CONSTRUIR", "CRIAR", "SOBREVIVER", "AUTOMATIZAR", "FARMAR"],
    },
    "VALORANT": {
        "context": ["MIRA", "RANKED", "AGENTES", "ELO", "HEADSHOT", "COMPETITIVO"],
        "objeto": ["MIRA", "POSICIONAMENTO", "CROSSHAIR", "ECONOMIA", "AGENTE"],
        "plataforma": ["VALORANT", "PC"],
        "acao": ["MELHORAR", "SUBIR", "DOMINAR", "TREINAR", "OTIMIZAR"],
    },
    "ACADEMIA": {
        "context": ["TREINO", "HIPERTROFIA", "DIETA", "FORCA", "MASSA MUSCULAR"],
        "objeto": ["TREINO", "DIETA", "PEITO", "COSTAS", "PERNA", "BICEPS"],
        "plataforma": ["ACADEMIA", "CASA", "PARQUE"],
        "acao": ["GANHAR", "DEFINIR", "AUMENTAR", "MELHORAR", "TREINAR"],
    },
    "CULINARIA": {
        "context": ["RECEITA", "RAPIDO", "FACIL", "FORNO", "FRIGIDEIRA", "DOCE"],
        "objeto": ["BOLO", "FRANGO", "PAO", "SOBREMESA", "MASSA", "MOLHO"],
        "plataforma": ["FORNO", "FOGAO", "AIRFRYER", "MICROONDAS"],
        "acao": ["PREPARAR", "FAZER", "COZINHAR", "ASSAR", "TEMPERAR"],
    },
    "FINANCAS": {
        "context": ["INVESTIR", "DINHEIRO", "RENDA", "ECONOMIZAR", "BOLSA", "CRIPTO"],
        "objeto": ["DINHEIRO", "RENDA EXTRA", "INVESTIMENTO", "ORCAMENTO", "POUPANCA"],
        "plataforma": ["BOLSA", "TESOURO", "BANCO", "CORRETORA"],
        "acao": ["INVESTIR", "ECONOMIZAR", "MULTIPLICAR", "POUPAR", "GANHAR"],
    },
    "VIAGEM": {
        "context": ["BARATO", "ROTEIRO", "DICAS", "MOCHILAO", "DESTINO", "PASSAGEM"],
        "objeto": ["VIAGEM", "ROTEIRO", "MALA", "PASSAGEM", "HOSPEDAGEM"],
        "plataforma": ["EUROPA", "BRASIL", "ASIA", "EUA"],
        "acao": ["VIAJAR", "ECONOMIZAR", "PLANEJAR", "EXPLORAR", "CONHECER"],
    },
    "ESTUDOS": {
        "context": ["CONCENTRACAO", "PROVA", "MEMORIZAR", "ENEM", "ROTINA", "FOCO"],
        "objeto": ["ESTUDO", "MEMORIA", "CONCENTRACAO", "PRODUTIVIDADE", "ROTINA"],
        "plataforma": ["ENEM", "VESTIBULAR", "CONCURSO", "FACULDADE"],
        "acao": ["MELHORAR", "AUMENTAR", "TURBINAR", "OTIMIZAR", "ORGANIZAR"],
    },
    "PROGRAMACAO": {
        "context": ["PYTHON", "CODIGO", "PROJETO", "CARREIRA", "DEV", "BACKEND"],
        "objeto": ["CODIGO", "PROJETO", "API", "APLICATIVO", "ALGORITMO", "SISTEMA"],
        "plataforma": ["PYTHON", "JAVASCRIPT", "WEB", "BACKEND"],
        "acao": ["APRENDER", "CRIAR", "DOMINAR", "DESENVOLVER", "OTIMIZAR"],
    },
    "MARKETING": {
        "context": ["VENDAS", "TRAFEGO", "INSTAGRAM", "CONVERSAO", "ANUNCIO", "FUNIL"],
        "objeto": ["VENDAS", "TRAFEGO", "ENGAJAMENTO", "AUDIENCIA", "MARCA"],
        "plataforma": ["INSTAGRAM", "TIKTOK", "YOUTUBE", "GOOGLE"],
        "acao": ["AUMENTAR", "ESCALAR", "TURBINAR", "MULTIPLICAR", "ATRAIR"],
    },
}

NUMS = ["3", "5", "7", "10", "12", "15", "20"]
YEARS = ["2024", "2025", "2026"]

# Templates grouped by intent category (documentation 4.4.2).
TEMPLATES = {
    "TUTORIAL": [
        "COMO {ACAO} SUA {OBJETO} NO {PLATAFORMA}",
        "COMO {ACAO} {OBJETO} DO ZERO EM {ANO}",
        "APRENDA A {ACAO} {OBJETO} RAPIDO",
        "PASSO A PASSO PARA {ACAO} {OBJETO}",
        "COMO {ACAO} {OBJETO} MESMO SENDO INICIANTE",
    ],
    "LISTA": [
        "{NUM} DICAS PARA {ACAO} SUA {OBJETO}",
        "{NUM} FORMAS DE {ACAO} {OBJETO} EM {ANO}",
        "{NUM} ERROS QUE TE IMPEDEM DE {ACAO} {OBJETO}",
        "{NUM} MELHORES DICAS DE {OBJETO} NO {PLATAFORMA}",
        "TOP {NUM} SEGREDOS PARA {ACAO} {OBJETO}",
    ],
    "GUIA": [
        "GUIA COMPLETO DE {OBJETO} NO {PLATAFORMA}",
        "GUIA DEFINITIVO PARA {ACAO} {OBJETO}",
        "TUDO SOBRE {OBJETO} QUE VOCE PRECISA SABER",
        "O GUIA COMPLETO DE {OBJETO} EM {ANO}",
    ],
    "COMPARATIVO": [
        "{OBJETO} VS {OBJETO2} O QUE E MELHOR PARA {PLATAFORMA}",
        "MELHOR FORMA DE {ACAO} {OBJETO} {OBJETO2} OU AMBOS",
        "{OBJETO} OU {OBJETO2} QUAL ESCOLHER PARA {ACAO}",
    ],
    "CLICKBAIT": [
        "VOCE NUNCA VAI {ACAO} {OBJETO} SEM SABER DISSO",
        "O SEGREDO QUE NINGUEM TE CONTOU PARA {ACAO} {OBJETO}",
        "ISSO VAI MUDAR COMO VOCE {ACAO} SUA {OBJETO}",
        "O METODO INACREDITAVEL PARA {ACAO} {OBJETO} EM {ANO}",
    ],
}


def _fill(template: str, voc: Dict[str, List[str]], rng: random.Random) -> str:
    obj_pool = voc["objeto"][:]
    rng.shuffle(obj_pool)
    obj = obj_pool[0]
    obj2 = obj_pool[1] if len(obj_pool) > 1 else obj
    return (template
            .replace("{ACAO}", rng.choice(voc["acao"]))
            .replace("{OBJETO2}", obj2)
            .replace("{OBJETO}", obj)
            .replace("{PLATAFORMA}", rng.choice(voc["plataforma"]))
            .replace("{NUM}", rng.choice(NUMS))
            .replace("{ANO}", rng.choice(YEARS)))


def generate(count: int, seed: int = 7) -> List[str]:
    rng = random.Random(seed)
    themes = list(THEMES.keys())
    intents = list(TEMPLATES.keys())
    per_theme = max(1, count // len(themes))

    blocks: List[str] = []
    seen_targets = set()
    for theme in themes:
        voc = THEMES[theme]
        made = 0
        attempts = 0
        while made < per_theme and attempts < per_theme * 10:
            attempts += 1
            intent = rng.choice(intents)
            template = rng.choice(TEMPLATES[intent])
            target = _fill(template, voc, rng)
            if target in seen_targets:
                continue
            seen_targets.add(target)
            ctx_pool = voc["context"][:]
            rng.shuffle(ctx_pool)
            ctx = ctx_pool[: rng.randint(3, min(6, len(ctx_pool)))]
            weight = rng.choice([1.0, 1.0, 1.0, 1.5, 2.0])
            blocks.append(
                "[SAMPLE]\n"
                f"theme={theme}\n"
                f"context={'|'.join(ctx)}\n"
                f"target={target}\n"
                f"weight={weight}\n"
                "[/SAMPLE]"
            )
            made += 1
    rng.shuffle(blocks)
    return blocks


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a .sds2 dataset for SSLE-1")
    ap.add_argument("--count", type=int, default=6000, help="approx number of samples")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", default="data/dataset.sds2")
    args = ap.parse_args()

    blocks = generate(args.count, args.seed)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks) + "\n")
    print(f"Wrote {len(blocks)} samples to {args.output}")


if __name__ == "__main__":
    main()
