#!/usr/bin/env python3
"""brand_probe — votre marque est-elle dans les poids d'un modèle de langage ?

Script AUTONOME (aucune dépendance sur le dépôt de la conférence) qui applique
la méthode présentée à TeknSEO, « GEO : le vrai du faux, mesuré par les
log-probs », à une marque de votre choix :

  1. PRÉSENCE dans le modèle — la marge de catégorie : le modèle préfère-t-il
     « Ahrefs is a software tool for search engine optimization » à
     « Ahrefs is a brand of Italian coffee » ? On mesure la même marge sur des
     noms inventés (leurres) dans le même run : c'est le plancher. Une marque
     sous le plancher est indistinguable d'un nom qui n'existe pas.

  2. TERMES ASSOCIÉS DANS LE MODÈLE — après « Ahrefs is a », quels mots le
     modèle privilégie-t-il PAR RAPPORT à ce qu'il dit d'un nom inventé ?
     Cette correction retire les mots que le modèle met après n'importe quel
     nom (« new », « popular », « company »).

  3. PRÉSENCE ET TERMES DANS LES CORPUS DE PRÉ-ENTRAÎNEMENT — via l'API
     publique infini-gram : occurrences dans C4, DCLM et Dolma, part des
     occurrences qui sont en réalité un autre mot (« Brevo » ⊂ « Brevoort »),
     et mots qui accompagnent la marque dans les documents.

Exemples :

    python brand_probe.py "Ahrefs" --category seo
    python brand_probe.py "Groupama" "MAIF" --category assurance --lang fr
    python brand_probe.py "Ma Marque" --category " is a software tool for rank tracking"
    python brand_probe.py "Ahrefs" --models qwen-1.5b llama-1b gemma-2b --json out.json
    python brand_probe.py --list-categories

⚠️ Ce que ce script NE mesure PAS : la citation par ChatGPT ou Gemini. Sur
156 marques et 1 100 réponses, la corrélation entre présence dans les poids
et fréquence de citation est de +0,1. Être dans les poids est une condition
d'entrée, pas une cause de citation. Voir README.md.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field

# ==========================================================================
# 1. Le protocole — figé, identique à celui des verdicts de la conférence
# ==========================================================================

# Modèles ouverts étudiés. `revision` est le commit HuggingFace épinglé : c'est
# ce qui rend la mesure reproductible d'une machine à l'autre. `gated` = il
# faut un compte HuggingFace et accepter la licence sur la page du modèle.
MODELS: dict[str, dict] = {
    # --- tier « scène » : tournent sur un portable, CPU, 8 Go de RAM -------
    "qwen-1.5b": {"hf_id": "Qwen/Qwen2.5-1.5B", "params_b": 1.54, "gated": False,
                  "revision": "8faed761d45a263340a0528343f099c05c9a4323"},
    "llama-1b": {"hf_id": "meta-llama/Llama-3.2-1B", "params_b": 1.24, "gated": True,
                 "revision": "4e20de362430cd3b72f300e6b0f18e50e7166e08"},
    "gemma-2b": {"hf_id": "google/gemma-2-2b", "params_b": 2.61, "gated": True,
                 "revision": "c5ebcd40d208330abc697524c919956e692655cf"},
    # --- tier « cloud » : GPU 24 Go ou plus (voir README) --------------------
    "qwen-7b": {"hf_id": "Qwen/Qwen2.5-7B", "params_b": 7.62, "gated": False,
                "revision": "d149729398750b98c0af14eb82c78cfe92750796"},
    "llama-8b": {"hf_id": "meta-llama/Llama-3.1-8B", "params_b": 8.03, "gated": True,
                 "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"},
    "gemma-9b": {"hf_id": "google/gemma-2-9b", "params_b": 9.24, "gated": True,
                 "revision": "33c193028431c2fde6c6e51f29e6f17b60cbfac6"},
}
STAGE_MODELS = ["qwen-1.5b", "llama-1b", "gemma-2b"]
CLOUD_MODELS = ["qwen-7b", "llama-8b", "gemma-9b"]

# Catégories : la phrase que le modèle doit préférer. Onze secteurs, en
# anglais (langue du panel) et en français (langue des prompts de citation).
# Une catégorie libre est acceptée : n'importe quelle phrase commençant par
# « is a… » ou « est un… ». La marque est collée devant.
CATEGORIES: dict[str, dict[str, str]] = {
    "en": {
        "seo": " is a software tool for search engine optimization",
        "analytics": " is a software tool for website analytics",
        "email": " is a software platform for email marketing",
        "support": " is a software platform for customer support",
        "survey": " is a software tool for creating online surveys",
        "social": " is a software platform for social media management",
        "crm": " is a software platform for customer relationship management",
        "assurance": " is an insurance company",
        "cosmetique": " is a cosmetics and skincare brand",
        "ameublement": " is a furniture and home decor retailer",
        "bricolage": " is a DIY and home improvement retailer",
    },
    "fr": {
        "seo": " est un logiciel de référencement naturel",
        "analytics": " est un outil d'analyse d'audience web",
        "email": " est une plateforme d'emailing",
        "support": " est un logiciel de support client",
        "survey": " est un outil de sondages en ligne",
        "social": " est un outil de gestion des réseaux sociaux",
        "crm": " est un logiciel de gestion de la relation client",
        "assurance": " est une compagnie d'assurance",
        "cosmetique": " est une marque de cosmétiques",
        "ameublement": " est une enseigne de meubles et de décoration",
        "bricolage": " est une enseigne de bricolage",
    },
}

# Fausses catégories, éloignées du domaine et sans rapport entre elles : si le
# modèle ne sait rien de la marque, aucune ne domine et la marge est ≈ 0.
WRONG_CATEGORIES: dict[str, list[str]] = {
    "en": [
        " is a brand of Italian coffee",
        " is a species of freshwater fish",
        " is a small town in southern France",
        " is a type of traditional folk dance",
    ],
    "fr": [
        " est une marque de café italien",
        " est une espèce de poisson d'eau douce",
        " est un petit village du sud de la France",
        " est une danse folklorique traditionnelle",
    ],
}

# Leurres : noms inventés de toutes pièces, mesurés DANS LE MÊME RUN que votre
# marque. Ils fixent le zéro de l'échelle empiriquement au lieu de le supposer.
DECOYS = ["Zorblax", "Vlurptix", "Quandelor", "Krendalis"]

# Leurres PARLANTS : un nom inventé par catégorie, qui annonce son métier
# (« Assurvex », « Mailzephor »). Ils répondent à l'objection la plus sérieuse
# qu'on puisse faire à la mesure : « votre marge ne fait que décoder le nom ».
# L'écart entre un leurre parlant et les leurres opaques chiffre ce que le nom
# seul rapporte sur ce modèle — jusqu'à +1 nat dans le talk. Mesuré sur
# qwen-1.5b, sondes fr : Assurvex +0,62 contre Groupama +0,42.
TRANSPARENT_DECOYS = {
    "seo": "Rankzuri", "analytics": "Analytivox", "email": "Mailzephor",
    "support": "Helpdesquo", "survey": "Surveymorph", "social": "Socialgrimp",
    "crm": "Clientarvo", "assurance": "Assurvex", "cosmetique": "Dermalixe",
    "ameublement": "Meublora", "bricolage": "Brikolan",
}

# Amorces pour les termes associés côté modèle.
TERM_TEMPLATES: dict[str, list[str]] = {
    "en": ["{} is a", "{} is known for its"],
    "fr": ["{} est un", "{} est connu pour ses"],
}

# Index infini-gram interrogés, avec ce qu'il faut savoir pour lire un chiffre.
INFINIGRAM_API = "https://api.infini-gram.io/"
INDEXES: dict[str, dict] = {
    "v4_c4train_llama": {"label": "C4", "note": "Common Crawl nettoyé, avril 2019"},
    "v4_dclm-baseline_llama": {"label": "DCLM", "note": "Common Crawl filtré, 2023"},
    "v4_dolma-v1_7_llama": {"label": "Dolma", "note": "web + Reddit + wiki + livres, début 2024"},
}
# Panier de normalisation : on compte des mots très communs dans chaque index
# et on s'en sert de dénominateur, plutôt que de faire confiance aux tailles
# de corpus publiées.
BASKET = ("the", "and", "of", "to", "in")
API_TIMEOUT_S = 40.0
API_MIN_INTERVAL_S = 0.25  # service public gratuit : on ne le martèle pas

STOPWORDS = set("""
the a an and or of to in for on with is are was were be been being it its this
that these those as at by from you your yours we our ours us i me my they them
their he she his her him not but if can will would could should may might must
have has had do does did done more most one two all also which what when where
how why than so about into out up down very just get got use using used like new
there here some any each other others such only own same then now over under
again further once because while before after above below between through during
without within off too s t don than
le la les l un une des du de d et ou en dans pour sur avec est sont était
étaient être été ce cet cette ces il elle ils elles nous vous je tu on ne pas plus
que qui quoi dont où ses son sa leur leurs mon ma mes ton ta tes notre nos votre
vos au aux par comme mais donc car ni si très aussi tout tous toute toutes même
fait faire peut pouvez avoir ont a y se sa lui eux moi toi cela ça
""".split())

MARGIN_UNCERTAIN = 0.10  # zone grise au-dessus du plancher, en nats


# ==========================================================================
# 2. Résultats
# ==========================================================================


@dataclass
class ModelResult:
    model: str
    lang: str
    category: str
    margin: float
    floor: float  # marge maximale des leurres, dans ce run
    decoy_margins: dict[str, float]
    verdict: str  # présente / incertaine / absente
    terms: list[dict] = field(default_factory=list)  # [{term, lift, prob}]
    terms_note: str = ""
    seconds: float = 0.0
    transparent_decoy: str | None = None  # nom inventé qui annonce la catégorie
    transparent_margin: float | None = None  # ce que le nom seul rapporte
    name_note: str = ""


@dataclass
class CorpusResult:
    query: str
    counts: dict[str, int]  # par index, occurrences de la SÉQUENCE DE TOKENS
    rates: dict[str, float]  # par index, occurrences par milliard de mots du panier
    glued_share: float | None  # part des occurrences qui continuent en un autre mot
    glued_example: str | None  # ex. « Brevo|ort »
    terms: list[dict] = field(default_factory=list)  # [{term, docs}]
    docs_sampled: int = 0
    note: str = ""


@dataclass
class BrandReport:
    brand: str
    category_key: str
    lang: str
    models: list[ModelResult] = field(default_factory=list)
    corpus: CorpusResult | None = None
    generated_at: str = ""


# ==========================================================================
# 3. Côté modèle — log-probabilités, teacher forcing, aucun échantillonnage
# ==========================================================================


class Scorer:
    """Moteur de log-probs minimal. Déterministe : mêmes poids, mêmes chiffres."""

    def __init__(self, model_key: str, device: str | None = None):
        if model_key not in MODELS:
            raise KeyError(f"Modèle inconnu : {model_key}. Connus : {', '.join(MODELS)}")
        self.key = model_key
        self.spec = MODELS[model_key]
        self._device = device
        self.model = None
        self.tokenizer = None

    @property
    def device(self):
        import torch

        if self._device:
            return torch.device(self._device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _dtype(self):
        import torch

        if self.device.type == "cuda":
            major, _ = torch.cuda.get_device_capability()
            # bf16 exige Ampere (sm_80) ou mieux ; sinon fp16 — sauf Gemma,
            # entraîné en bf16, qui déborde en fp16 : voir README.
            return torch.bfloat16 if major >= 8 else torch.float16
        try:
            if torch.backends.cpu.get_cpu_capability() in ("AVX512_BF16", "AMX"):
                return torch.bfloat16
        except (AttributeError, RuntimeError):
            pass
        return torch.float32

    def load(self) -> "Scorer":
        if self.model is not None:
            return self
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kw = {"revision": self.spec["revision"]}
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec["hf_id"], **kw)
        dtype = self._dtype()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.spec["hf_id"], dtype=dtype, **kw)
        except TypeError:  # transformers < 4.56
            self.model = AutoModelForCausalLM.from_pretrained(
                self.spec["hf_id"], torch_dtype=dtype, **kw)
        self.model.to(self.device)
        self.model.eval()
        return self

    def unload(self) -> None:
        import torch

        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- primitive : log-prob moyenne d'une continuation sachant un préfixe --

    def _encode_pair(self, prefix: str, continuation: str) -> tuple[list[int], int]:
        # On tokenise le texte complet d'un bloc, jamais les deux morceaux
        # séparément : les tokenizers BPE fusionnent à travers la frontière.
        if prefix.endswith(" ") and not continuation.startswith(" "):
            prefix, continuation = prefix.rstrip(" "), " " + continuation
        p_ids = self.tokenizer(prefix, add_special_tokens=True)["input_ids"]
        f_ids = self.tokenizer(prefix + continuation, add_special_tokens=True)["input_ids"]
        if f_ids[: len(p_ids)] != p_ids:
            raise RuntimeError(
                f"Frontière de token instable entre {prefix!r} et {continuation!r}.")
        if len(f_ids) <= len(p_ids):
            raise RuntimeError(f"La continuation {continuation!r} n'ajoute aucun token.")
        return f_ids, len(p_ids)

    def mean_logprob(self, prefix: str, continuation: str) -> float:
        import torch

        self.load()
        ids, start = self._encode_pair(prefix, continuation)
        x = torch.tensor([ids], device=self.device)
        with torch.inference_mode():
            logits = self.model(x).logits
        lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        tgt = x[0, 1:]
        total = sum(float(lp[pos, int(tgt[pos])]) for pos in range(start - 1, len(ids) - 1))
        return total / (len(ids) - start)

    def category_margin(self, name: str, category: str, wrong: list[str]) -> float:
        """Marge de catégorie : log-prob moyenne de la bonne phrase moins celle
        de la meilleure fausse phrase. > 0 = le modèle préfère la vraie."""
        good = self.mean_logprob(name, category)
        best_wrong = max(self.mean_logprob(name, w) for w in wrong)
        return good - best_wrong

    def next_token_logprobs(self, prefix: str):
        """Vecteur complet de log-probs du token suivant (taille du vocabulaire)."""
        import torch

        self.load()
        ids = self.tokenizer(prefix, add_special_tokens=True)["input_ids"]
        x = torch.tensor([ids], device=self.device)
        with torch.inference_mode():
            logits = self.model(x).logits
        return torch.log_softmax(logits[0, -1].float(), dim=-1)


_TOKEN_WORD = re.compile(r"^ ?[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ-]{2,}$")
_WORD_PIECE = re.compile(r"^[A-Za-zÀ-ÿ-]+$")  # suite de mot : lettres, sans espace


def term_lifts(brand_lp, decoy_lps: list, decode, k: int = 200,
               min_prob: float = 0.005) -> list[dict]:
    """Termes que le modèle met après la marque PLUS que derrière un nom inventé.

    `lift` = log p(token | marque) − moyenne des log p(token | leurre), en nats.
    On ne garde que les tokens que le modèle juge au moins un peu probables
    pour la marque (≥ 0,5 %), puis on classe par lift. Un lift de +7 nats veut
    dire « mille fois plus probable après votre marque qu'après un nom inventé ».
    """
    import torch

    top = torch.topk(brand_lp, min(k, brand_lp.numel()))
    base = torch.stack(decoy_lps).mean(dim=0)
    out = []
    for lp, i in zip(top.values.tolist(), top.indices.tolist()):
        if lp < math.log(min_prob):
            break
        tok = decode([i])
        if not _TOKEN_WORD.match(tok):
            continue
        out.append({"term": tok.strip(), "prob": round(math.exp(lp), 4),
                    "lift": round(lp - float(base[i]), 2), "token_id": i})
    out.sort(key=lambda d: -d["lift"])
    return out


def complete_word(scorer: Scorer, prefix: str, token_id: int, max_extra: int = 3) -> str:
    """Prolonge un token jusqu'à la fin du mot, en glouton.

    Les tokenizers coupent les mots français en morceaux (« synd|icat »,
    « parten|aire ») : le lift est calculé sur le premier morceau, mais on
    affiche le mot entier, tant que le token suivant le plus probable est une
    suite de lettres sans espace.
    """
    import torch

    ids = scorer.tokenizer(prefix, add_special_tokens=True)["input_ids"] + [token_id]
    word = scorer.tokenizer.decode([token_id]).strip()
    for _ in range(max_extra):
        x = torch.tensor([ids], device=scorer.device)
        with torch.inference_mode():
            logits = scorer.model(x).logits
        nxt = int(torch.argmax(logits[0, -1]))
        piece = scorer.tokenizer.decode([nxt])
        if not _WORD_PIECE.match(piece):
            break
        word += piece
        ids.append(nxt)
    return word


def verdict_of(margin: float, floor: float) -> str:
    if margin > floor + MARGIN_UNCERTAIN:
        return "présente"
    if margin > floor:
        return "incertaine"
    return "absente"


def measure_model(scorer: Scorer, brand: str, lang: str, category: str,
                  top: int = 12, decoy_cache: dict | None = None,
                  transparent: str | None = None) -> ModelResult:
    t0 = time.monotonic()
    wrong = WRONG_CATEGORIES[lang]
    decoy_cache = decoy_cache if decoy_cache is not None else {}
    key = (scorer.key, lang, category)
    if key not in decoy_cache:
        decoy_cache[key] = {d: scorer.category_margin(d, category, wrong) for d in DECOYS}
        if transparent:
            decoy_cache[key][transparent] = scorer.category_margin(transparent, category, wrong)
    decoy_margins = dict(decoy_cache[key])
    t_margin = decoy_margins.pop(transparent, None) if transparent else None
    floor = max(decoy_margins.values())
    margin = scorer.category_margin(brand, category, wrong)
    verdict = verdict_of(margin, floor)

    name_note = ""
    if t_margin is not None and verdict != "absente" and margin <= t_margin:
        name_note = (f"marge inférieure à celle d'un nom inventé qui annonce la catégorie "
                     f"({transparent} : {t_margin:+.2f}) : peut n'être qu'un effet de nom")

    terms: list[dict] = []
    note = ""
    if verdict == "absente":
        note = ("termes non calculés : sous le plancher des noms inventés, "
                "les associations ne seraient que du bruit")
    else:
        merged: dict[str, dict] = {}
        for tmpl in TERM_TEMPLATES[lang]:
            prefix = tmpl.format(brand)
            brand_lp = scorer.next_token_logprobs(prefix)
            decoy_lps = [scorer.next_token_logprobs(tmpl.format(d)) for d in DECOYS]
            for row in term_lifts(brand_lp, decoy_lps, scorer.tokenizer.decode)[:top]:
                row["term"] = complete_word(scorer, prefix, row.pop("token_id"))
                cur = merged.get(row["term"].lower())
                if cur is None or row["lift"] > cur["lift"]:
                    merged[row["term"].lower()] = row
        terms = sorted(merged.values(), key=lambda d: -d["lift"])[:top]
        if verdict == "incertaine":
            note = "marque dans la zone grise : termes à lire avec prudence"
    return ModelResult(scorer.key, lang, category, round(margin, 3), round(floor, 3),
                       {k: round(v, 3) for k, v in decoy_margins.items()}, verdict,
                       terms, note, round(time.monotonic() - t0, 1),
                       transparent, None if t_margin is None else round(t_margin, 3),
                       name_note)


# ==========================================================================
# 4. Côté corpus — infini-gram (API publique, gratuite, sans garantie)
# ==========================================================================


class InfiniGram:
    def __init__(self):
        self._last = 0.0
        self._basket: dict[str, float] = {}

    def post(self, payload: dict) -> dict:
        err: Exception | None = None
        for attempt in range(4):
            wait = API_MIN_INTERVAL_S - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            req = urllib.request.Request(
                INFINIGRAM_API, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=API_TIMEOUT_S) as r:
                    data = json.loads(r.read())
                self._last = time.monotonic()
                if isinstance(data, dict) and data.get("error"):
                    raise RuntimeError(f"infini-gram : {data['error']} ({payload})")
                return data
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                err = exc
                self._last = time.monotonic()
                time.sleep(2 ** attempt)
        raise RuntimeError(f"infini-gram injoignable après 4 essais : {err}")

    def count(self, index: str, query: str) -> int:
        return int(self.post({"index": index, "query_type": "count", "query": query})["count"])

    def basket_total(self, index: str) -> float:
        if index not in self._basket:
            self._basket[index] = float(sum(self.count(index, w) for w in BASKET))
        return self._basket[index]

    def rate(self, index: str, query: str) -> float:
        return self.count(index, query) / self.basket_total(index) * 1e9

    def next_tokens(self, index: str, query: str) -> dict:
        return self.post({"index": index, "query_type": "ntd", "query": query})

    def search_docs(self, index: str, query: str, maxnum: int = 10,
                    max_disp_len: int = 400) -> dict:
        return self.post({"index": index, "query_type": "search_docs", "query": query,
                          "maxnum": maxnum, "max_disp_len": max_disp_len})


def glued_share(ntd: dict) -> tuple[float | None, str | None]:
    """Part des occurrences où la marque est suivie d'une CONTINUATION DE MOT.

    infini-gram apparie des séquences de tokens, pas des mots : « Brevo »
    matche « Brevoort ». Dans le tokenizer Llama, un token qui commence un
    nouveau mot porte le marqueur « ▁ » ; un token qui commence par une
    lettre nue colle au précédent. On renvoie la part de ces continuations
    et la plus fréquente (ex. « ort »).
    """
    res = ntd.get("result_by_token_id") or {}
    total = sum(v["cont_cnt"] for v in res.values())
    if not total:
        return None, None
    glued = {v["token"]: v["cont_cnt"] for v in res.values() if v["token"][:1].isalpha()}
    share = sum(glued.values()) / total
    example = max(glued, key=glued.get) if glued else None
    return share, example


_WORD = re.compile(r"[A-Za-zÀ-ÿ][A-Za-z0-9À-ÿ'’-]{2,}")


def cooccurrences(docs: list[dict], brand: str) -> collections.Counter:
    """Fréquence documentaire des mots dans les extraits contenant la marque."""
    skip = {w.lower() for w in re.findall(_WORD, brand)} | {brand.lower()}
    c: collections.Counter = collections.Counter()
    for doc in docs:
        text = "".join(span[0] for span in doc.get("spans", []))
        words = {w.lower().strip("'’-") for w in _WORD.findall(text)}
        c.update(w for w in words if w not in STOPWORDS and w not in skip and len(w) > 2)
    return c


def measure_corpus(ig: InfiniGram, brand: str, query: str | None = None,
                   top: int = 20, pages: int = 5, log=None) -> CorpusResult:
    query = query or brand
    counts, rates = {}, {}
    for idx in INDEXES:
        counts[idx] = ig.count(idx, query)
        rates[idx] = round(counts[idx] / ig.basket_total(idx) * 1e9, 2)
        if log:
            log(f"    {INDEXES[idx]['label']:6s} {counts[idx]:>10,d} occurrences")

    share, example = None, None
    if " AND " not in query and " OR " not in query:
        # Le collage se mesure sur l'index le plus gros où la marque apparaît.
        best = max(INDEXES, key=lambda i: counts[i])
        if counts[best]:
            share, example = glued_share(ig.next_tokens(best, query))
            if example:
                example = f"{query}|{example}"

    note = ""
    terms: list[dict] = []
    sampled = 0
    if share is not None and share > 0.5:
        note = (f"{share:.0%} des occurrences sont un autre mot ({example}) : "
                "comptages non fiables, termes non calculés. Précisez la requête "
                "avec --corpus-query (ex. \"Brevo AND emailing\").")
    else:
        seen: set = set()
        docs: list[dict] = []
        for idx in INDEXES:
            if not counts[idx]:
                continue
            for _ in range(pages):
                d = ig.search_docs(idx, query)
                for i, doc in zip(d.get("idxs", []), d.get("documents", [])):
                    if (idx, i) not in seen:
                        seen.add((idx, i))
                        docs.append(doc)
                if d.get("cnt", 0) <= 10:
                    break
        sampled = len(docs)
        terms = [{"term": w, "docs": n} for w, n in cooccurrences(docs, brand).most_common(top)]
        if share is not None and share > 0.1:
            note = f"{share:.0%} des occurrences sont un autre mot ({example}) : comptages surestimés d'autant"
    return CorpusResult(query, counts, rates, None if share is None else round(share, 3),
                        example, terms, sampled, note)


# ==========================================================================
# 5. Orchestration et rendu
# ==========================================================================


def resolve_category(spec: str, lang: str) -> tuple[str, str]:
    """`seo` → clé + phrase ; une phrase libre (« is a … ») → clé « custom »."""
    if spec in CATEGORIES[lang]:
        return spec, CATEGORIES[lang][spec]
    s = spec.strip()
    if s.lower().startswith(("is ", "est ", "was ", "était ")) or s.startswith(" "):
        return "custom", " " + s.lstrip()
    raise ValueError(
        f"Catégorie inconnue : {spec!r}. Utilisez une clé ({', '.join(CATEGORIES[lang])}) "
        "ou une phrase libre commençant par « is a … » / « est un … ».")


def probe(brands: list[str], category: str, lang: str = "en",
          models: list[str] | None = None, corpus: bool = True,
          corpus_query: str | None = None, top: int = 12, log=None) -> list[BrandReport]:
    log = log or (lambda *_: None)
    models = ["qwen-1.5b"] if models is None else models  # [] = aucun modèle
    cat_key, cat_sentence = resolve_category(category, lang)
    reports = {b: BrandReport(b, cat_key, lang, generated_at=time.strftime("%Y-%m-%d %H:%M"))
               for b in brands}

    for mk in models:
        spec = MODELS[mk]
        log(f"\n[{mk}] chargement de {spec['hf_id']} ({spec['params_b']} Md)"
            f"{' — modèle à licence, compte HuggingFace requis' if spec['gated'] else ''}…")
        t0 = time.monotonic()
        scorer = Scorer(mk).load()
        log(f"[{mk}] chargé en {time.monotonic() - t0:.0f} s sur {scorer.device}")
        cache: dict = {}
        for b in brands:
            r = measure_model(scorer, b, lang, cat_sentence, top=top, decoy_cache=cache,
                              transparent=TRANSPARENT_DECOYS.get(cat_key))
            reports[b].models.append(r)
            log(f"[{mk}] {b}: marge {r.margin:+.2f}, plancher {r.floor:+.2f} → {r.verdict} "
                f"({r.seconds} s)")
        scorer.unload()

    if corpus:
        ig = InfiniGram()
        for b in brands:
            log(f"\n[corpus] {b} ({corpus_query or b})")
            try:
                reports[b].corpus = measure_corpus(ig, b, corpus_query, top=top, log=log)
            except RuntimeError as exc:
                log(f"[corpus] {exc}")
                reports[b].corpus = CorpusResult(corpus_query or b, {}, {}, None, None,
                                                 note=f"indisponible : {exc}")
    return list(reports.values())


def render(report: BrandReport) -> str:
    L = ["", "=" * 72, f"  {report.brand}  —  catégorie « {report.category_key} », sondes {report.lang}",
         "=" * 72]
    if report.models:
        present = sum(1 for m in report.models if m.verdict == "présente")
        L += ["", f"  PRÉSENCE DANS LES MODÈLES : {present}/{len(report.models)} "
              f"au-dessus du plancher des noms inventés", ""]
        L.append(f"  {'modèle':10s} {'marge':>7s} {'plancher':>9s} {'nom parlant':>12s}  verdict")
        for m in report.models:
            tm = f"{m.transparent_margin:>+12.2f}" if m.transparent_margin is not None else f"{'—':>12s}"
            L.append(f"  {m.model:10s} {m.margin:>+7.2f} {m.floor:>+9.2f} {tm}  {m.verdict}")
        t = next((m.transparent_decoy for m in report.models if m.transparent_decoy), None)
        if t:
            L.append(f"  (nom parlant = « {t} », inventé, qui annonce la catégorie : "
                     "ce que le nom seul rapporte)")
        for m in report.models:
            L.append("")
            if m.name_note:
                L.append(f"  ⚠ {m.model} : {m.name_note}")
            if m.terms:
                L.append(f"  Termes associés dans {m.model} (lift en nats face aux noms inventés) :")
                L.append("    " + ", ".join(f"{t['term']} (+{t['lift']:.1f})" for t in m.terms))
            if m.terms_note:
                L.append(f"    ({m.terms_note})")
    c = report.corpus
    if c is not None:
        L += ["", "  CORPUS DE PRÉ-ENTRAÎNEMENT (infini-gram) :", ""]
        if c.counts:
            L.append(f"  {'index':7s} {'occurrences':>12s} {'par milliard':>13s}   contenu")
            for idx, n in c.counts.items():
                L.append(f"  {INDEXES[idx]['label']:7s} {n:>12,d} {c.rates[idx]:>13.1f}   "
                         f"{INDEXES[idx]['note']}")
        if c.note:
            L += ["", f"  ⚠ {c.note}"]
        if c.terms:
            L += ["", f"  Mots qui accompagnent « {c.query} » ({c.docs_sampled} extraits, "
                  "nombre d'extraits) :"]
            L.append("    " + ", ".join(f"{t['term']} ({t['docs']})" for t in c.terms))
    L += ["", "  Lecture : la présence dans les poids est une condition d'entrée, pas une",
          "  cause de citation (corrélation présence ↔ citation mesurée : +0,1).", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("brands", nargs="*", help="une ou plusieurs marques")
    ap.add_argument("--category", "-c", default="seo",
                    help="clé de catégorie (voir --list-categories) ou phrase libre « is a … »")
    ap.add_argument("--lang", "-l", choices=["en", "fr"], default="en",
                    help="langue des sondes (en = langue du panel de la conférence)")
    ap.add_argument("--models", "-m", nargs="+", default=["qwen-1.5b"],
                    help=f"parmi {', '.join(MODELS)} (défaut : qwen-1.5b, sans licence)")
    ap.add_argument("--no-model", action="store_true", help="corpus seulement")
    ap.add_argument("--no-corpus", action="store_true", help="modèles seulement (hors ligne)")
    ap.add_argument("--corpus-query", help="requête infini-gram à la place du nom, "
                    "syntaxe « A AND B » pour désambiguïser")
    ap.add_argument("--top", type=int, default=12, help="nombre de termes affichés")
    ap.add_argument("--json", help="écrire le rapport complet dans ce fichier")
    ap.add_argument("--list-categories", action="store_true")
    args = ap.parse_args(argv)

    if args.list_categories:
        for lang, cats in CATEGORIES.items():
            print(f"\n[{lang}]")
            for k, v in cats.items():
                print(f"  {k:12s} « <marque>{v} »")
        return 0
    if not args.brands:
        ap.error("indiquez au moins une marque")
    for mk in args.models:
        if mk not in MODELS:
            ap.error(f"modèle inconnu : {mk}. Connus : {', '.join(MODELS)}")
    try:
        resolve_category(args.category, args.lang)
    except ValueError as exc:
        ap.error(str(exc))

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    reports = probe(args.brands, args.category, args.lang,
                    models=[] if args.no_model else args.models,
                    corpus=not args.no_corpus, corpus_query=args.corpus_query,
                    top=args.top, log=log)
    for r in reports:
        print(render(r))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in reports], f, ensure_ascii=False, indent=2)
        log(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    try:
        import signal

        signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # `| head` sans traceback
    except (AttributeError, ValueError):  # Windows
        pass
    raise SystemExit(main())
