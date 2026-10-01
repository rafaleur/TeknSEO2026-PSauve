"""brand_probe_app — l'interface du kit : la même mesure, à l'écran.

    streamlit run brand_probe_app.py

Plusieurs marques — la vôtre et ses concurrentes —, une catégorie, un ou
plusieurs modèles, et le rapport de `brand_probe.py` dessiné : par modèle, le
classement des marques sur la marge de catégorie, face au nom parlant, au
meilleur nom inventé et au plancher ; par marque, les termes que le modèle
lui associe, sa fréquence dans les corpus de pré-entraînement et les mots qui
l'y accompagnent. Le calcul est celui du script, à l'identique ; l'interface
n'ajoute rien à la méthode. Les modèles restent chargés entre deux mesures.

La charte est celle de la démo de scène de la conférence : fond blanc, encre
noire, Figtree, angles vifs ; ce qu'on mesure en cyan plein, ce qu'on lui
compare (nom inventé, plancher) au trait, barre creuse — l'opposition tient
au remplissage autant qu'à la couleur. Le thème est dans
`.streamlit/config.toml`, la police dans `static/`.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from html import escape
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import brand_probe as bp  # noqa: E402

# --------------------------------------------------------------------------
# Charte : la palette des slides et de la démo de scène
# --------------------------------------------------------------------------

SURFACE = "#ffffff"
TEXT_PRIMARY = "#111110"
TEXT_SECONDARY = "#54544d"
TEXT_MUTED = "#8b8b83"
GRID = "#d7d7d0"
ACCENT = "#23bced"  # cyan — ce qu'on mesure
ACCENT_2 = "#03c245"  # vert — la concordance, le bon signe
FONT = "Figtree, Segoe UI, system-ui, sans-serif"

# Tailles en pixels dans le SVG : le dessin se met à l'échelle de la colonne
# (viewBox), donc ces valeurs sont relatives à une largeur de 1100.
FS_VALUE = 30
FS_LABEL = 26
FS_AXIS = 22

MASTHEAD = "Pierre Sauvé"
MASTHEAD_ROLE = "Consultant SEO · GEO · Formateur"
MASTHEAD_EVENT = "TEKNSEO · 2026"
MASTHEAD_FOOT = "TeknSEO · Conférence de Pierre Sauvé · Theblackroom.io · Kit de démo"

CSS = f"""
<style>
  [data-testid="stToolbar"], [data-testid="stDecoration"],
  [data-testid="stStatusWidget"], footer {{ display: none !important; }}
  [data-testid="stHeader"] {{ background: transparent; height: 0; }}
  .block-container {{ padding-top: 2rem; }}
  h1 {{ font-size: 2.6rem !important; line-height: 1.1;
        font-weight: 800; letter-spacing: -0.025em; }}
  h2 {{ font-size: 1.9rem !important; margin-top: 1.2rem;
        font-weight: 700; letter-spacing: -0.015em; }}
  h3 {{ font-weight: 700; letter-spacing: -0.01em; }}
  p, li, label, .stMarkdown {{ font-size: 1.2rem !important; line-height: 1.5; }}
  .verdict {{ font-size: 1.45rem; line-height: 1.45; font-weight: 600;
              border-left: 8px solid {TEXT_PRIMARY};
              padding: 0.6rem 0 0.6rem 1.2rem; margin: 1rem 0; }}
  .limite {{ font-size: 1.1rem; line-height: 1.5;
             border-left: 8px solid {GRID}; color: {TEXT_SECONDARY};
             padding: 0.5rem 0 0.5rem 1.2rem; margin: 0.8rem 0; }}
  .muted {{ color: {TEXT_MUTED}; font-size: 1rem !important; }}
  .stButton button {{ font-size: 1.2rem; padding: 0.5rem 1.4rem; font-weight: 600; }}
  section[data-testid="stSidebar"] {{ min-width: 21rem; }}
  .masthead {{ display: flex; align-items: stretch; margin-bottom: 1.6rem;
               border: 2px solid {TEXT_PRIMARY}; }}
  .masthead .who {{ padding: 0.45rem 1.1rem; font-size: 1.05rem;
                    white-space: nowrap; align-self: center; }}
  .masthead .who b {{ font-weight: 800; }}
  .masthead .who span {{ color: {TEXT_SECONDARY}; margin-left: 0.8rem; }}
  .masthead .gap {{ flex: 1; border-left: 2px solid {TEXT_PRIMARY}; }}
  .masthead .event {{ background: {TEXT_PRIMARY}; color: #ffffff;
                      padding: 0.45rem 1.2rem; font-size: 0.95rem; font-weight: 700;
                      letter-spacing: 0.2em; white-space: nowrap; display: flex;
                      align-items: center; }}
  .eyebrow {{ font-size: 0.95rem !important; letter-spacing: 0.24em;
              text-transform: uppercase; color: {TEXT_MUTED}; margin: 0 0 0.1rem; }}
  .footrule {{ border-top: 2px solid {TEXT_PRIMARY}; margin-top: 3rem;
               padding-top: 0.6rem; font-size: 0.9rem; letter-spacing: 0.2em;
               text-transform: uppercase; color: {TEXT_MUTED}; }}
  table {{ font-size: 1.05rem; }}
</style>
"""


# --------------------------------------------------------------------------
# Graphiques SVG — ceux de la démo de scène, à l'identique
# --------------------------------------------------------------------------


def _text(x, y, content, size=FS_LABEL, fill=TEXT_PRIMARY, anchor="start", weight="400"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" font-family="{FONT}">'
            f'{escape(str(content))}</text>')


def _bar(x, y, w, h, hollow: bool = False, colour: str = ACCENT) -> str:
    """Pleine, en accent, pour ce qu'on mesure ; creuse pour la référence."""
    if hollow:
        return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 3):.1f}" height="{h}" '
                f'fill="{SURFACE}" stroke="{TEXT_PRIMARY}" stroke-width="3"/>')
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h}" fill="{colour}"/>'


def _frame(width: int, height: int, body: str) -> str:
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" '
            f'style="max-width:{width}px;display:block;margin:0 auto" '
            f'xmlns="http://www.w3.org/2000/svg" role="img">'
            f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>{body}</svg>')


def probe_ranking(rows: list[dict], key: str, title: str, floor: float | None = None,
                  width: int = 1100, left: int = 380, hollow_names: set | None = None) -> str:
    """Classement sur la marge de catégorie : barres depuis le zéro, le signe
    se lit d'un coup d'œil. Sous le plancher des noms inventés, la barre se
    vide ; les références (noms inventés) sont creuses aussi."""
    if not rows:
        return ""
    ordered = sorted(rows, key=lambda r: -r[key])
    hollow_names = hollow_names or set()
    bar_h, gap, top = 64, 30, 84
    height = top + len(ordered) * (bar_h + gap) + 24
    plot_w = width - left - 200
    values = [r[key] for r in ordered] + ([floor] if floor is not None else [])
    lo, hi = min(min(values), 0.0), max(max(values), 0.0)
    span = (hi - lo) or 1
    zero = left + (0 - lo) / span * plot_w
    parts = [_text(left, 44, title, FS_AXIS, TEXT_MUTED),
             f'<line x1="{zero:.1f}" y1="{top - 14}" x2="{zero:.1f}" y2="{height - 14}" '
             f'stroke="{GRID}" stroke-width="2"/>']
    if floor is not None:
        fx = left + (floor - lo) / span * plot_w
        parts.append(f'<line x1="{fx:.1f}" y1="{top - 14}" x2="{fx:.1f}" y2="{height - 14}" '
                     f'stroke="{TEXT_PRIMARY}" stroke-width="2" stroke-dasharray="8 8"/>')
        parts.append(_text(fx + 8, top - 22, f"plancher {floor:+.2f}", FS_AXIS, TEXT_SECONDARY))
    for i, r in enumerate(ordered):
        y = top + i * (bar_h + gap)
        value = r[key]
        w = max(4, abs(value) / span * plot_w)
        x = zero if value >= 0 else zero - w
        under = floor is not None and value < floor
        parts.append(_bar(x, y, w, bar_h, hollow=under or r["name"] in hollow_names))
        parts.append(_text(left - 32, y + bar_h * 0.52, r["name"], 30, TEXT_PRIMARY, "end", "600"))
        vx = x + w + 22 if value >= 0 else zero + 22
        parts.append(_text(vx, y + bar_h * 0.62, f"{value:+.2f}", FS_VALUE, TEXT_PRIMARY, "start", "600"))
        if under:
            parts.append(_text(left - 32, y + bar_h * 0.92, "indistinguable d'un nom inventé",
                               FS_AXIS, TEXT_MUTED, "end"))
    return _frame(width, height, "".join(parts))


def term_bars(terms: list[dict], title: str, key: str = "lift", width: int = 1100) -> str:
    """Les termes associés à une marque — barres compactes, une par terme."""
    if not terms:
        return ""
    ordered = sorted(terms, key=lambda t: -t[key])
    bar_h, gap, left, top = 40, 12, 380, 76
    height = top + len(ordered) * (bar_h + gap) + 20
    plot_w = width - left - 200
    hi = max(t[key] for t in ordered) or 1
    parts = [_text(left, 44, title, FS_AXIS, TEXT_MUTED),
             f'<line x1="{left}" y1="{top - 10}" x2="{left}" y2="{height - 12}" '
             f'stroke="{GRID}" stroke-width="2"/>']
    for i, t in enumerate(ordered):
        y = top + i * (bar_h + gap)
        w = max(4, t[key] / hi * plot_w)
        parts.append(_bar(left, y, w, bar_h))
        parts.append(_text(left - 28, y + bar_h * 0.72, t["term"], 28, TEXT_PRIMARY, "end", "600"))
        label = f"{t[key]:+.1f}" if key == "lift" else f"{t[key]}"
        parts.append(_text(left + w + 18, y + bar_h * 0.72, label, FS_AXIS, TEXT_PRIMARY, "start", "600"))
    return _frame(width, height, "".join(parts))


def rate_bars(c: bp.CorpusResult, width: int = 1100) -> str:
    """Occurrences par milliard de mots, un index par barre ; creuses si le
    comptage est approché."""
    rows = [{"term": bp.INDEXES[i]["label"], "v": r} for i, r in c.rates.items()]
    if not rows:
        return ""
    hi = max(r["v"] for r in rows) or 1
    bar_h, gap, left, top = 40, 12, 380, 76
    height = top + len(rows) * (bar_h + gap) + 20
    plot_w = width - left - 200
    parts = [_text(left, 44, "occurrences par milliard de mots"
                   + (" (≈ comptage approché)" if c.approx else ""), FS_AXIS, TEXT_MUTED)]
    for i, r in enumerate(rows):
        y = top + i * (bar_h + gap)
        w = max(4, r["v"] / hi * plot_w)
        parts.append(_bar(left, y, w, bar_h, hollow=c.approx))
        parts.append(_text(left - 28, y + bar_h * 0.72, r["term"], 28, TEXT_PRIMARY, "end", "600"))
        parts.append(_text(left + w + 18, y + bar_h * 0.72, f"{r['v']:.1f}", FS_AXIS,
                           TEXT_PRIMARY, "start", "600"))
    return _frame(width, height, "".join(parts))


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
# Rendu
# --------------------------------------------------------------------------


def _html(markup: str) -> None:
    if markup:
        st.markdown(markup, unsafe_allow_html=True)


def _masthead() -> None:
    _html(f'<div class="masthead"><div class="who"><b>{MASTHEAD}</b>'
          f'<span>{MASTHEAD_ROLE}</span></div><div class="gap"></div>'
          f'<div class="event">{MASTHEAD_EVENT}</div></div>')


def _title(eyebrow: str, title: str) -> None:
    _html(f'<p class="eyebrow">{eyebrow}</p>')
    st.title(title)


def verdict_line(brand: str, m: bp.ModelResult) -> str:
    if m.verdict == "présente":
        return (f"{brand} est <b>présente</b> dans {m.model} : marge {m.margin:+.2f}, "
                f"plancher des noms inventés {m.floor:+.2f}.")
    if m.verdict == "incertaine":
        return (f"{brand} est dans la <b>zone grise</b> de {m.model} : marge {m.margin:+.2f}, "
                f"plancher {m.floor:+.2f}, à moins de 0,10 nat.")
    return (f"{brand} est <b>indistinguable d'un nom inventé</b> pour {m.model} : "
            f"marge {m.margin:+.2f}, plancher {m.floor:+.2f}.")


def show_model_ranking(mk: str, reports: list[bp.BrandReport]) -> None:
    """Un modèle, toutes les marques : le classement, puis les verdicts."""
    results = [(r.brand, m) for r in reports for m in r.models if m.model == mk]
    if not results:
        return
    spec = bp.MODELS.get(mk, {})
    st.header(f"{mk} · {spec.get('params_b', '?')} Md")
    first = results[0][1]
    rows = [{"name": b, "margin": m.margin} for b, m in results]
    hollow = set()
    worst = max(first.decoy_margins, key=first.decoy_margins.get)
    rows.append({"name": f"{worst} · inventé", "margin": first.decoy_margins[worst]})
    hollow.add(f"{worst} · inventé")
    if first.transparent_margin is not None:
        name = f"{first.transparent_decoy} · inventé, parlant"
        rows.append({"name": name, "margin": first.transparent_margin})
        hollow.add(name)
    _html(probe_ranking(rows, "margin", "marge de catégorie, en nats — plus à droite, mieux connue",
                        floor=first.floor, hollow_names=hollow))
    for b, m in results:
        _html(f'<div class="verdict">{verdict_line(b, m)}</div>')
        if m.name_note:
            _html(f'<div class="limite">EFFET DU NOM — {m.name_note}.</div>')


def show_brand_detail(r: bp.BrandReport) -> None:
    """Une marque : ses termes par modèle, puis les corpus."""
    for m in r.models:
        st.subheader(f"Ce que {m.model} associe à « {r.brand} »")
        if m.terms:
            _html(term_bars(m.terms, "lift en nats face aux noms inventés"))
            _html('<p class="muted">+7 nats ≈ mille fois plus probable après la marque '
                  "qu'après un nom inventé.</p>")
        elif m.terms_note:
            _html(f'<div class="limite">{m.terms_note}.</div>')
    c = r.corpus
    if c is not None:
        st.subheader(f"« {r.brand} » dans les corpus de pré-entraînement")
        if c.counts:
            head = "| index | occurrences | par milliard de mots |"
            sep = "|---|---:|---:|"
            if c.whole_word:
                head += " en mot entier (≈) |"
                sep += "---:|"
            head += " contenu |"
            sep += "---|"
            lines = [head, sep]
            for idx, n in c.counts.items():
                cell = f"{'≈' if c.approx else ''}{n:,d}".replace(",", " ")
                line = f"| {bp.INDEXES[idx]['label']} | {cell} | {c.rates[idx]:.1f} |"
                if c.whole_word:
                    line += f" {c.whole_word.get(idx, n):,d} |".replace(",", " ")
                lines.append(line + f" {bp.INDEXES[idx]['note']} |")
            st.markdown("\n".join(lines))
            if c.approx:
                _html('<p class="muted">≈ comptage approché : documents où les termes voisinent '
                      "à moins de 1 000 tokens, clauses fréquentes sous-échantillonnées par l'API.</p>")
            _html(rate_bars(c))
        if c.note:
            _html(f'<div class="limite">{c.note.replace("--corpus-query", "le champ « Requête corpus »")}</div>')
        if c.terms:
            _html(term_bars(c.terms, f"mots qui accompagnent « {c.query} », sur {c.docs_sampled} extraits",
                            key="docs"))
    with st.expander(f"Rapport texte et JSON · {r.brand}"):
        st.code(bp.render(r), language=None)
        st.download_button("Télécharger le JSON", json.dumps(asdict(r), ensure_ascii=False, indent=2),
                           file_name=f"brand_probe_{r.brand}.json", mime="application/json",
                           key=f"dl-{r.brand}")


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="brand_probe — kit de démo TeknSEO", layout="wide")
    _html(CSS)

    sb = st.sidebar
    sb.header("Mesurer")
    raw = sb.text_area("Marques (une par ligne) — la vôtre et ses concurrentes",
                       "Screaming Frog\nOncrawl\nBotify", height=120, key="brands")
    brands = [b.strip() for b in raw.splitlines() if b.strip()]
    lang = sb.radio("Sondes", ["en", "fr"], horizontal=True, key="lang",
                    format_func=lambda x: {"en": "anglais (langue du panel)", "fr": "français"}[x])
    keys = list(bp.CATEGORIES[lang])
    cat_key = sb.selectbox("Catégorie", keys + ["phrase libre"], index=keys.index("seo"), key="category")
    if cat_key == "phrase libre":
        category = sb.text_input("« <marque> … »", "is a software tool for rank tracking" if lang == "en"
                                 else "est un logiciel de suivi de positionnement", key="free")
    else:
        category = cat_key
    sb.markdown(f'<p class="muted">« &lt;marque&gt;'
                f'{escape(bp.CATEGORIES[lang].get(cat_key, " " + category.strip()))} »</p>',
                unsafe_allow_html=True)

    def _model_label(mk: str) -> str:
        s = bp.MODELS[mk]
        tier = "GPU 24 Go" if s["params_b"] > 5 else "CPU"
        return f"{mk} · {s['params_b']} Md · {tier}{' · licence HF' if s['gated'] else ''}"

    models = sb.multiselect("Modèles", list(bp.MODELS), default=["qwen-1.5b"], key="models",
                            format_func=_model_label)
    corpus = sb.checkbox("Interroger les corpus (réseau, API infini-gram)", value=True, key="corpus")
    corpus_query = sb.text_input("Requête corpus (optionnel)", "", key="corpus_query",
                                 help="Pour un homonyme : « MAIF AND assurance ». Ne corrige pas un "
                                      "collage de tokens. S'applique à chaque marque.").strip() or None
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

    _masthead()
    _title("Kit de démo de la conférence", "Votre marque, dans les poids")

    if not go:
        st.markdown("Plusieurs marques, une catégorie, et ce que les poids en savent : la marge de "
                    "catégorie face au plancher des noms inventés, les termes associés, la fréquence "
                    "dans C4, DCLM et Dolma. Laquelle le modèle connaît-il le mieux ?")
        for b in brands:
            st.markdown(f"### {b}")
        _html('<div class="limite">Note : ce kit mesure la présence dans les poids, pas la citation '
              "par ChatGPT ou Gemini : corrélation mesurée entre les deux, +0,1. Être dans les poids "
              "est une condition d'entrée, pas une cause.</div>")
        _html(f'<div class="footrule">{MASTHEAD_FOOT}</div>')
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

    _html(f'<p class="muted">« &lt;marque&gt;{escape(sentence)} » contre quatre fausses catégories, '
          "et quatre noms inventés mesurés en même temps — sondes "
          f"{'en anglais' if lang == 'en' else 'en français'}.</p>")

    lines: list[str] = []
    t0 = time.monotonic()
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
        status.update(label=f"Mesure terminée en {time.monotonic() - t0:.0f} s",
                      state="complete", expanded=False)

    # 1. Par modèle : le classement des marques, puis les verdicts.
    for mk in models:
        show_model_ranking(mk, reports)

    # 2. Par marque : les termes, les corpus. Dépliés s'il n'y a qu'une marque.
    st.header("Marque par marque")
    for r in reports:
        if len(reports) == 1:
            show_brand_detail(r)
        else:
            with st.expander(r.brand, expanded=False):
                show_brand_detail(r)

    _html('<p class="muted">Le même code que brand_probe.py · la présence ne prédit pas la citation '
          "(ρ ≈ +0,1).</p>")
    _html(f'<div class="footrule">{MASTHEAD_FOOT}</div>')


if __name__ == "__main__":
    main()
