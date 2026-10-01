"""brand_probe_app — l'interface du kit : la même mesure, à l'écran.

    pip install streamlit
    streamlit run brand_probe_app.py

Un formulaire (marques, catégorie, langue, modèles, corpus), un bouton, et
le rapport de `brand_probe.py` en graphiques : la marge de la marque face au
plancher des noms inventés, les termes que le modèle lui associe, la
fréquence dans les corpus de pré-entraînement et les mots qui l'y
accompagnent. Le calcul est celui du script, à l'identique ; l'interface
n'ajoute rien à la méthode. Les modèles restent chargés entre deux mesures.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import brand_probe as bp  # noqa: E402

# Palette : la marque en bleu, le nom parlant en orange, les noms inventés en
# gris ; le plancher est un trait. Les verdicts portent un signe et un mot,
# jamais la couleur seule.
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#a3a29c", "#52514e"
VERDICT = {"présente": ("✓", "#008300"), "incertaine": ("~", "#b87a00"), "absente": ("✕", "#c93c3b")}


# --------------------------------------------------------------------------
# Ressources gardées entre deux mesures
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _scorer(model_key: str) -> "bp.Scorer":
    return bp.Scorer(model_key).load()


@st.cache_resource(show_spinner=False)
def _decoy_cache() -> dict:
    """Les leurres d'une catégorie ne se mesurent qu'une fois par modèle."""
    return {}


@st.cache_resource(show_spinner=False)
def _infinigram() -> "bp.InfiniGram":
    return bp.InfiniGram()


def run(brands: list[str], category: str, lang: str, models: list[str], corpus: bool,
        corpus_query: str | None, log) -> list[bp.BrandReport]:
    """`bp.probe`, sans décharger les modèles entre deux clics."""
    cat_key, sentence = bp.resolve_category(category, lang)
    reports = {b: bp.BrandReport(b, cat_key, lang, generated_at=time.strftime("%Y-%m-%d %H:%M"))
               for b in brands}
    for mk in models:
        spec = bp.MODELS[mk]
        t0 = time.monotonic()
        log(f"[{mk}] chargement de {spec['hf_id']} ({spec['params_b']} Md)…")
        scorer = _scorer(mk)
        log(f"[{mk}] prêt en {time.monotonic() - t0:.0f} s sur {scorer.device}")
        for b in brands:
            r = bp.measure_model(scorer, b, lang, sentence, decoy_cache=_decoy_cache(),
                                 transparent=bp.TRANSPARENT_DECOYS.get(cat_key))
            reports[b].models.append(r)
            log(f"[{mk}] {b} : marge {r.margin:+.2f}, plancher {r.floor:+.2f} → {r.verdict}")
    if corpus:
        ig = _infinigram()
        for b in brands:
            log(f"[corpus] {b} ({corpus_query or b})")
            try:
                reports[b].corpus = bp.measure_corpus(ig, b, corpus_query, log=log)
            except RuntimeError as exc:
                log(f"[corpus] {exc}")
                reports[b].corpus = bp.CorpusResult(corpus_query or b, {}, {}, None, None,
                                                    note=f"indisponible : {exc}")
    return list(reports.values())


# --------------------------------------------------------------------------
# Graphiques
# --------------------------------------------------------------------------


def _bars(rows: list[dict], x: str, y: str, x_title: str, color: str | None = None,
          domain: list[str] | None = None, colors: list[str] | None = None,
          fmt: str = "+.2f", rule: float | None = None, rule_title: str = ""):
    """Barres horizontales, une ligne par entrée ; `rule` trace un repère vertical."""
    import altair as alt

    enc = {
        "y": alt.Y(f"{y}:N", sort=None, title=None, axis=alt.Axis(labelLimit=280, labelOverlap=False)),
        "x": alt.X(f"{x}:Q", title=x_title),
        "tooltip": [alt.Tooltip(f"{y}:N", title=" "), alt.Tooltip(f"{x}:Q", title=x_title, format=fmt)],
    }
    if color:
        enc["color"] = alt.Color(f"{color}:N", scale=alt.Scale(domain=domain, range=colors),
                                 legend=alt.Legend(title=None, orient="bottom"))
    else:
        enc["color"] = alt.value(BLUE)
    base = alt.Chart(alt.Data(values=rows))
    bars = base.mark_bar(size=14, cornerRadiusEnd=4).encode(**enc)
    if rule is None:
        chart = bars
    else:
        line = base.mark_rule(strokeDash=[4, 4], color=INK, size=2).encode(
            x=alt.datum(rule), tooltip=alt.value(f"{rule_title} : {rule:+.2f}"))
        chart = alt.layer(bars, line)
    # Streamlit ajuste le graphique entier (axes et légende compris) à cette
    # hauteur : 28 px par ligne, plus l'axe des x et, s'il y a une légende,
    # sa ligne, sinon six étiquettes se chevauchent et disparaissent.
    return chart.properties(height=28 * len(rows) + (95 if color else 55))


def margin_chart(m: bp.ModelResult, brand: str):
    rows = [{"nom": brand, "marge": round(m.margin, 2), "rôle": "la marque"}]
    if m.transparent_decoy and m.transparent_margin is not None:
        rows.append({"nom": f"{m.transparent_decoy} (inventé, annonce la catégorie)",
                     "marge": round(m.transparent_margin, 2), "rôle": "nom parlant"})
    rows += [{"nom": f"{d} (inventé)", "marge": round(v, 2), "rôle": "nom inventé"}
             for d, v in m.decoy_margins.items()]
    return _bars(rows, "marge", "nom", "marge de catégorie (nats)", color="rôle",
                 domain=["la marque", "nom parlant", "nom inventé"], colors=[BLUE, ORANGE, GRAY],
                 rule=round(m.floor, 2), rule_title="plancher des noms inventés")


def terms_chart(terms: list[dict], key: str, title: str, fmt: str = "+.1f"):
    rows = [{"terme": t["term"], "valeur": round(t[key], 2)} for t in terms]
    return _bars(rows, "valeur", "terme", title, fmt=fmt)


# --------------------------------------------------------------------------
# Rendu d'un rapport
# --------------------------------------------------------------------------


def _verdict(v: str) -> str:
    sign, colour = VERDICT.get(v, ("·", INK))
    return f"<span style='color:{colour};font-weight:700'>{sign} {v}</span>"


def show_model(m: bp.ModelResult, brand: str) -> None:
    spec = bp.MODELS.get(m.model, {})
    st.markdown(f"**{m.model}** · {spec.get('params_b', '?')} Md · verdict : {_verdict(m.verdict)}",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Marge de la marque", f"{m.margin:+.2f} nat")
    c2.metric("Plancher des noms inventés", f"{m.floor:+.2f} nat",
              help="La marge la plus haute obtenue par quatre noms inventés, dans ce run, sur ce modèle.")
    if m.transparent_margin is not None:
        c3.metric(f"Nom parlant ({m.transparent_decoy})", f"{m.transparent_margin:+.2f} nat",
                  help="Un nom inventé qui annonce sa catégorie : ce que le nom seul rapporte.")
    if m.name_note:
        st.warning(m.name_note)
    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(margin_chart(m, brand), width="stretch")
    with right:
        if m.terms:
            st.caption(f"Termes que {m.model} associe à « {brand} » — lift en nats face aux noms inventés")
            st.altair_chart(terms_chart(m.terms, "lift", "lift (nats)"), width="stretch")
        elif m.terms_note:
            st.caption(m.terms_note)


def show_corpus(c: bp.CorpusResult) -> None:
    st.markdown("**Corpus de pré-entraînement** (infini-gram)")
    if c.counts:
        rows = []
        for idx, n in c.counts.items():
            row = {"index": bp.INDEXES[idx]["label"], "contenu": bp.INDEXES[idx]["note"],
                   "occurrences": f"{'≈' if c.approx else ''}{n:,d}".replace(",", " "),
                   "par milliard de mots": round(c.rates.get(idx, 0.0), 1)}
            if c.whole_word:
                row["en mot entier (≈)"] = f"{c.whole_word.get(idx, n):,d}".replace(",", " ")
            rows.append(row)
        left, right = st.columns([3, 2])
        with left:
            st.dataframe(rows, hide_index=True, width="stretch")
            if c.approx:
                st.caption("≈ comptage approché : documents où les termes voisinent à moins de "
                           "1 000 tokens, clauses fréquentes sous-échantillonnées par l'API.")
        with right:
            chart_rows = [{"index": bp.INDEXES[i]["label"], "par milliard": round(r, 1)}
                          for i, r in c.rates.items()]
            st.altair_chart(_bars(chart_rows, "par milliard", "index", "occurrences par milliard de mots",
                                  fmt=".1f"), width="stretch")
    if c.note:
        (st.warning if c.glued_share and c.glued_share > 0.5 else st.info)(c.note)
    if c.terms:
        st.caption(f"Mots qui accompagnent « {c.query} » ({c.docs_sampled} extraits, nombre d'extraits)")
        st.altair_chart(terms_chart(c.terms, "docs", "extraits", fmt="d"), width="stretch")


def show_report(r: bp.BrandReport, sentence: str) -> None:
    st.markdown(f"## {r.brand}")
    st.caption(f"« {r.brand}{sentence} » — sondes {'en anglais' if r.lang == 'en' else 'en français'}")
    if r.models:
        n_ok = sum(m.verdict == "présente" for m in r.models)
        st.markdown(f"**Présence dans les modèles : {n_ok}/{len(r.models)}** au-dessus du plancher des noms inventés")
        for m in r.models:
            show_model(m, r.brand)
    if r.corpus is not None:
        show_corpus(r.corpus)
    with st.expander("Rapport texte et JSON"):
        st.code(bp.render(r), language=None)
        st.download_button("Télécharger le JSON", json.dumps(asdict(r), ensure_ascii=False, indent=2),
                           file_name=f"brand_probe_{r.brand}.json", mime="application/json",
                           key=f"dl-{r.brand}")


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="brand_probe", page_icon="🔎", layout="wide")
    st.title("brand_probe — votre marque est-elle dans les poids des modèles ?")
    st.caption("Kit à emporter de la conférence TeknSEO « GEO : le vrai du faux, mesuré par les log-probs ». "
               "Ce kit mesure la présence dans les poids, pas la citation par ChatGPT ou Gemini : "
               "corrélation mesurée entre les deux, +0,1.")

    sb = st.sidebar
    sb.header("Mesurer")
    brands_raw = sb.text_input("Marques (séparées par des virgules)", "Ahrefs", key="brands")
    lang = sb.radio("Sondes", ["en", "fr"], horizontal=True, key="lang",
                    format_func=lambda x: {"en": "anglais (langue du panel)", "fr": "français"}[x])
    keys = list(bp.CATEGORIES[lang])
    cat_key = sb.selectbox("Catégorie", keys + ["phrase libre"], index=keys.index("seo"), key="category")
    if cat_key == "phrase libre":
        category = sb.text_input("« <marque> … »", "is a software tool for rank tracking" if lang == "en"
                                 else "est un logiciel de suivi de positionnement", key="free")
    else:
        category = cat_key
    sb.caption(f"« <marque>{bp.CATEGORIES[lang].get(cat_key, ' ' + category.strip())} »")

    def _model_label(mk: str) -> str:
        s = bp.MODELS[mk]
        tier = "GPU 24 Go" if s["params_b"] > 5 else "CPU"
        return f"{mk} · {s['params_b']} Md · {tier}{' · licence HF' if s['gated'] else ''}"

    models = sb.multiselect("Modèles", list(bp.MODELS), default=["qwen-1.5b"], key="models",
                            format_func=_model_label)
    corpus = sb.checkbox("Interroger les corpus (réseau, API infini-gram)", value=True, key="corpus")
    corpus_query = sb.text_input("Requête corpus (optionnel)", "", key="corpus_query",
                                 help="Pour un homonyme : « MAIF AND assurance ». "
                                      "Ne corrige pas un collage de tokens.").strip() or None
    if any(bp.MODELS[m]["gated"] for m in models) and not os.environ.get("HF_TOKEN"):
        token = sb.text_input("Jeton Hugging Face (Llama, Gemma)", "", type="password", key="hf_token",
                              help="Licence acceptée sur la page du modèle, jeton « Read » : voir README.")
        if token:
            os.environ["HF_TOKEN"] = token
    go = sb.button("Mesurer", type="primary", key="go")
    if sb.button("Libérer les modèles chargés", key="free_models"):
        _scorer.clear()
        _decoy_cache.clear()
        sb.caption("Mémoire libérée.")

    brands = [b.strip() for b in brands_raw.split(",") if b.strip()]
    if not go:
        st.markdown("Une marque, une catégorie, et ce que les poids en savent : la marge de catégorie "
                    "face au plancher des noms inventés, les termes associés, la fréquence dans C4, DCLM "
                    "et Dolma. Détails, pièges et lecture du rapport dans le `README.md` du kit.")
        return
    if not brands:
        st.error("Indiquez au moins une marque.")
        return
    if not models and not corpus:
        st.error("Choisissez au moins un modèle, ou cochez les corpus.")
        return
    try:
        _, sentence = bp.resolve_category(category, lang)
    except ValueError as exc:
        st.error(str(exc))
        return

    lines: list[str] = []
    with st.status("Mesure en cours…", expanded=True) as status:
        box = st.empty()

        def log(msg: str) -> None:
            lines.append(msg)
            box.code("\n".join(lines[-12:]), language=None)

        try:
            reports = run(brands, category, lang, models, corpus, corpus_query, log)
        except Exception as exc:  # modèle à licence refusé, réseau, mémoire…
            status.update(label="La mesure a échoué", state="error")
            st.error(f"{type(exc).__name__} : {exc}")
            return
        status.update(label="Mesure terminée", state="complete", expanded=False)

    if len(reports) > 1 and models:
        st.markdown("### Vue d'ensemble")
        rows = [{"marque": r.brand, **{m.model: f"{m.margin:+.2f} · {m.verdict}" for m in r.models}}
                for r in reports]
        st.dataframe(rows, hide_index=True, width="stretch")
    for r in reports:
        show_report(r, sentence)


if __name__ == "__main__":
    main()
