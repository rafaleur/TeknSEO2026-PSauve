# brand_probe — votre marque est-elle dans les poids des modèles ?

Kit à emporter de la conférence TeknSEO **« GEO : le vrai du faux, mesuré par
les log-probs »**. Un seul script, sans nos données ni nos verdicts, qui
applique la méthode du talk à *votre* marque — en ligne de commande, dans
Colab, ou dans une petite interface.

Dépôt : https://github.com/rafaleur/TeknSEO2026-PSauve — ou, sans rien installer,
[ouvrir le notebook dans Colab](https://colab.research.google.com/github/rafaleur/TeknSEO2026-PSauve/blob/main/brand_probe.ipynb).

| ce que vous obtenez | comment c'est mesuré |
|---|---|
| **Présence** de la marque dans chacun des modèles ouverts étudiés | marge de catégorie en log-probabilités, plancher fixé par des noms inventés mesurés dans le même run |
| **Termes** que le modèle associe à la marque | distribution du mot suivant après « *Marque* is a », corrigée de ce que le modèle dit d'un nom inventé |
| **Fréquence** dans les corpus de pré-entraînement | occurrences dans C4, DCLM et Dolma via l'API publique infini-gram, normalisées par milliard de mots |
| **Termes** qui accompagnent la marque sur le web d'entraînement | mots les plus fréquents dans les extraits de documents qui contiennent la marque |

> ⚠️ **Ce que le kit ne mesure pas : la citation par ChatGPT ou Gemini.**
> Sur 156 marques et 1 100 réponses, la corrélation entre présence dans les
> poids et fréquence de citation est de **+0,1**. Être dans les poids est une
> condition d'entrée, pas une cause de citation. Une marque absente des poids
> n'est jamais citée sans recherche web ; une marque présente n'est pas citée
> pour autant.

## Installation (5 minutes, CPU suffisant)

```bash
git clone https://github.com/rafaleur/TeknSEO2026-PSauve && cd TeknSEO2026-PSauve
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Premier lancement : le modèle par défaut, **Qwen 2.5 1.5B**, se télécharge
(3 Go), sans compte ni licence. Ensuite tout tourne hors ligne, sauf la partie
corpus.

## Utilisation

```bash
# Votre marque, dans sa catégorie (sondes en anglais, langue du panel du talk)
python brand_probe.py "Ahrefs" --category seo

# Plusieurs marques, sondes en français, catégorie grand public
python brand_probe.py "Groupama" "MAIF" "Wakam" --category assurance --lang fr

# Une catégorie qui n'est pas dans la liste : n'importe quelle phrase « is a … »
python brand_probe.py "Ma Marque" --category "is a software tool for rank tracking"
python brand_probe.py "Ma Marque" --category "est une banque en ligne" --lang fr

# Les trois modèles de scène, rapport complet en JSON
python brand_probe.py "Ahrefs" --models qwen-1.5b llama-1b gemma-2b --json ahrefs.json

# Corpus seulement (aucun modèle chargé) / modèles seulement (hors ligne)
python brand_probe.py "Ahrefs" --no-model
python brand_probe.py "Ahrefs" --no-corpus

python brand_probe.py --list-categories
```

Durées mesurées sans GPU, poids déjà téléchargés : chargement de qwen-1.5b
**30 à 220 s** selon le disque, puis, par marque, **15 à 20 s** sur un CPU à
bf16 natif (Ryzen AI, Core Ultra, Xeon récents : AVX-512 BF16 ou AMX) et
**50 à 90 s** sur un CPU AVX2 classique, où le script calcule en fp32 (mesuré
sur un Ryzen 7 3700X, 8 cœurs). La première marque est la plus longue : les
quatre leurres ne sont mesurés qu'une fois par modèle, quel que soit le nombre
de marques.

## L'interface

La même mesure, à l'écran, pour qui préfère un formulaire à une ligne de
commande :

```bash
streamlit run brand_probe_app.py
```

Le navigateur s'ouvre sur http://localhost:8501 : vos marques, une par
ligne — la vôtre et ses concurrentes —, la catégorie, la langue, les modèles,
les corpus, puis **Mesurer**. Pour chaque modèle, les marques sont classées
sur la marge de catégorie, face au nom parlant, au meilleur nom inventé et au
plancher : laquelle le modèle connaît-il le mieux ? Une barre pleine est une
marque présente ; sous le plancher, elle se vide. Puis, marque par marque,
les termes associés, la fréquence par corpus et les mots qui l'accompagnent ;
le rapport texte et le JSON sont dans un volet dépliable. Les modèles restent
chargés entre deux mesures, les leurres d'une catégorie ne sont mesurés
qu'une fois, et un bouton libère la mémoire. C'est `brand_probe.py` qui
calcule : l'interface n'ajoute rien à la méthode. La charte est celle de la
démo de scène : `.streamlit/config.toml` (thème) et `static/` (la police
Figtree, embarquée pour tourner hors ligne).

## Lire le rapport

```
  Ahrefs  —  catégorie « seo », sondes en

  PRÉSENCE DANS LES MODÈLES : 1/1 au-dessus du plancher des noms inventés

  modèle       marge  plancher  nom parlant  verdict
  qwen-1.5b    +2.32     -0.18        -0.52  présente
  (nom parlant = « Rankzuri », inventé, qui annonce la catégorie : ce que le nom seul rapporte)

  Termes associés dans qwen-1.5b (lift en nats face aux noms inventés) :
    keyword (+9.3), crawler (+8.8), SEO (+8.7), search (+6.7), Google (+5.3) …

  CORPUS DE PRÉ-ENTRAÎNEMENT (infini-gram) :
  index    occurrences  par milliard   contenu
  C4            18,309         960.5   Common Crawl nettoyé, avril 2019
  …
  Mots qui accompagnent « Ahrefs » (150 extraits, nombre d'extraits) :
    seo (87), google (77), search (77), tools (75), site (70) …
```

- **Marge** : log-probabilité moyenne de « *Marque* is a tool for search
  engine optimization » moins celle de la meilleure des quatre fausses
  phrases (« … is a brand of Italian coffee », « … a species of freshwater
  fish », …). En nats. Positif = le modèle préfère la vraie catégorie.
- **Plancher** : la marge la plus haute obtenue par quatre noms inventés
  (« Zorblax », « Vlurptix », …) *dans ce run, sur ce modèle*. Il est
  recalculé à chaque fois : si les leurres montent, l'instrument est cassé et
  vous le voyez.
- **Verdict** : *présente* si la marge dépasse le plancher de plus de 0,10
  nat ; *incertaine* entre les deux ; *absente* en dessous, c'est-à-dire
  indistinguable d'un nom qui n'existe pas. Les termes ne sont calculés que
  pour une marque présente : sous le plancher, ils ne seraient que du bruit
  (« revolutionary », « premium », « innovative » sortent derrière n'importe quel
  nom inconnu).
- **Nom parlant** : pour les onze catégories connues, le script mesure aussi un
  nom inventé qui *annonce* sa catégorie (« Rankzuri » pour le SEO,
  « Assurvex » pour l'assurance, « Mailzephor » pour l'emailing). Sa marge est
  ce que le nom seul rapporte sur ce modèle, sans aucune connaissance derrière.
  Elle n'entre pas dans le plancher, c'est une référence : une marque dont la
  marge ne dépasse pas celle du nom parlant est signalée, car sa présence
  peut n'être qu'un effet de nom. Mesuré sur qwen-1.5b, sondes en français :
  Assurvex +0,62, Groupama +0,42.
- **Lift** d'un terme : log p(terme | marque) − log p(terme | nom inventé).
  +7 nats = mille fois plus probable après votre marque qu'après un nom
  inventé. Un lift proche de 0 (« new », « company », « popular ») est ce que
  le modèle dit de tout le monde. Le lift est calculé sur le premier token du
  mot ; le mot est ensuite complété en glouton, parce que les tokenizers
  coupent le français en morceaux (« synd|icat », « parten|aire »).
- **Par milliard** : occurrences rapportées au nombre de « the, and, of, to,
  in » du même index, mesuré le même jour. C'est le seul moyen de comparer des
  corpus de tailles différentes sans se fier à des tailles publiées.

## Quatre pièges, tous rencontrés en construisant le talk

**1. Le collage de tokens.** infini-gram apparie des séquences de tokens, pas
des mots. « Brevo » compte 3 421 occurrences dans C4… dont 98 % sont
« Brevo**ort** » (un éditeur de Marvel) et « Brevo**ortia** » (un poisson).
Le script le détecte (distribution du token suivant), donne le comptage en
mot entier (Brevo seul : 53 dans C4, 228 dans DCLM, 621 dans Dolma) et,
au-delà de 50 %, refuse de calculer des termes : les extraits seraient ceux
de l'autre mot. Une requête « Brevo AND emailing » n'y change rien, nous
l'avons vérifié : « Brevoort » la satisfait aussi, et la centaine de
documents qu'elle renvoie parlent tous de l'éditeur.

Le remède par requête existe pour l'autre cas, l'**homonyme** entier :
« MAIF » dans les corpus anglophones est surtout le *Maryland Automobile
Insurance Fund* (les mots qui l'accompagnent le disent : « maryland »,
« drivers », « auto »). Une requête « A AND B » ne garde que les documents où
les deux termes voisinent :

```bash
python brand_probe.py "MAIF" --category assurance --lang fr --corpus-query "MAIF AND assurance"
```

Le AND déplace l'équilibre, il ne purifie pas : sur Dolma, 21 extraits sur
30 parlent alors de l'assureur, contre 1 sur 30 sans le AND, mais dans C4 et
DCLM, anglophones, le fonds du Maryland domine toujours et les termes le
montrent (« maryland », « auto »). Le comptage est marqué « ≈ » : l'API
compte des documents, pas des occurrences, à moins de 1 000 tokens d'écart,
et sous-échantillonne les clauses fréquentes. Il situe un ordre de grandeur,
il ne se compare pas au « par milliard » d'une marque non ambiguë. Dans tous
les cas, lisez les termes du corpus avant de croire le comptage.

**2. Les corpus sont datés.** C4 est un instantané d'avril 2019, DCLM de
2023, Dolma de début 2024. Une marque récente, ou rebaptisée (Sendinblue →
Brevo en 2023), n'y est pas ou presque. Ce n'est pas un bug du script :
c'est la réalité de ce sur quoi les modèles ont appris, et c'est ce qui
explique que Brevo soit sous le plancher des noms inventés pour Qwen 1.5B et
Llama 1B (les 7–9 Md, entraînés plus tard, la connaissent, faiblement : +0,4
à +0,8 nat) tout en étant cité dans 83 % des réponses de ChatGPT sur
l'emailing — la citation vient de la recherche web, pas des poids.

**3. Les petits modèles ne sont pas ChatGPT.** Les trois modèles de scène
(1 à 2,6 milliards de paramètres) connaissent moins de marques que les
7–9 milliards, qui en connaissent moins que les modèles fermés. Une marque
*absente* à 1,5 Md peut être présente à 9 Md. La direction des résultats du
talk (Wikipédia, Reddit, mentions) est la même aux deux tailles, mais le
niveau de présence de *votre* marque, lui, dépend de la taille. D'où l'option
GPU ci-dessous.

**4. Le nom parlant.** Un nom qui annonce son métier obtient une marge sans
que le modèle sache quoi que ce soit : « Assurvex », inventé, dépasse Groupama
sur qwen-1.5b. C'est pour cela que le script mesure le nom parlant de la
catégorie et vous prévient quand votre marge ne le dépasse pas. Pour une
catégorie libre, il n'y a pas de nom parlant : comparez vous-même à un nom
inventé de votre cru qui « sonne » comme votre secteur.

## Les modèles à licence : passer par Hugging Face

Qwen est libre. **Llama** (Meta) et **Gemma** (Google) exigent un compte
Hugging Face et l'acceptation de leur licence sur la page du modèle, sinon le
téléchargement échoue avec une erreur 401 ou 403.

1. Créer un compte sur https://huggingface.co et le confirmer par e-mail.
2. Ouvrir la page de chaque modèle voulu et cliquer sur *Agree and access
   repository* (l'accès Meta prend de quelques minutes à quelques heures) :
   - https://huggingface.co/meta-llama/Llama-3.2-1B
   - https://huggingface.co/google/gemma-2-2b
   - https://huggingface.co/meta-llama/Llama-3.1-8B et
     https://huggingface.co/google/gemma-2-9b pour le tier GPU
3. Créer un jeton d'accès (*Settings → Access Tokens*, type *Read* ou un jeton
   *fine-grained* avec « Read access to contents of all public gated repos you
   can access ») : https://huggingface.co/settings/tokens
4. Se connecter une fois sur la machine :

```bash
hf auth login            # huggingface_hub ≥ 0.34 ; sinon : huggingface-cli login
# ou, sans interaction (Colab, serveur) :
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

Le script épingle la **révision** exacte de chaque modèle (le commit
Hugging Face) : deux personnes, deux machines, deux dates donnent la même
marge à la décimale près. C'est ce qui manque aux modèles fermés, qui
changent sous vos pieds.

## Option : les modèles 7–9 milliards sur GPU

Les trois grands modèles étudiés dans le talk, **Qwen 2.5 7B, Llama 3.1 8B et
Gemma 2 9B**, demandent une carte NVIDIA avec **24 Go** de mémoire (les poids
en bf16 pèsent 15 à 19 Go). Trois façons d'en avoir une :

| option | carte | coût | remarque |
|---|---|---|---|
| Google Colab gratuit | T4, 16 Go | 0 € | suffit pour 1,5 et 3 Md ; **pas** pour 7–9 Md en bf16 |
| Google Colab Pro | L4 24 Go ou A100 40 Go | ≈ 10 €/mois | choisir *Runtime → L4* ou *A100* |
| Runpod, Vast.ai, Lambda… | A5000 / A6000 / A100 | ≈ 0,3 à 1,5 €/h | une mesure complète des trois modèles tient en 20 minutes |

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
export HF_TOKEN=hf_xxx           # licences Llama et Gemma acceptées au préalable
python brand_probe.py "Ahrefs" --models qwen-7b llama-8b gemma-9b --category seo
```

Deux précisions matérielles :

- **Gemma exige bf16**, donc une carte Ampere ou plus récente (A-series, L4,
  RTX 30xx/40xx). Sur une carte Turing (T4, RTX 20xx) le script bascule en
  fp16 et Gemma déborde : marges aberrantes. Utilisez Qwen et Llama sur ces
  cartes.
- **Pas de quantification.** Charger un modèle en 8 ou 4 bits change les
  log-probabilités, donc les marges, et casse la comparaison avec les chiffres
  du talk. Le script charge toujours les poids pleins.

Le notebook `brand_probe.ipynb` fait tout cela dans Colab, gratuit ou Pro,
en trois cellules.

## L'API infini-gram

La partie corpus interroge https://infini-gram.io (Liu et al., 2024), service
public et gratuit de l'Allen Institute for AI, sans clé. Le script attend
250 ms entre deux requêtes et réessaie quatre fois. Une marque coûte une
quarantaine de requêtes. Merci de ne pas lancer le kit à cent en même temps
pendant une conférence ; l'API n'a aucune garantie de disponibilité, et
`--no-corpus` fonctionne hors ligne.

## Reproduire les chiffres du talk

Ce kit est le sous-ensemble « une marque à la fois » de la méthode. Le dépôt
complet (panel de 156 marques, appariement sur la présence web, tests de
signe, validation contre 1 100 réponses ChatGPT et Gemini) est celui de la
conférence ; le kit reprend à l'identique ses catégories, ses fausses
catégories, ses leurres et ses révisions de modèles, et un test l'y vérifie.

Licence du kit : MIT (fichier `LICENSE`). Les modèles restent sous leurs licences respectives.
