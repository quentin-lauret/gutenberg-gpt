"""Nettoie les fichiers texte de data/books des phrases parasites (Gutenberg & co)."""

import argparse
import html
import re
import sys
from pathlib import Path

from paths import DATA_DIRECTORY

# --- Coupures de queue : tout ce qui suit le marqueur est supprime -------------
PG_END = re.compile(
    r"(\*\*\*\s*)?End of (the |this )?Project Gutenberg.*$"
    r"|\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*$"
    r"|Fin (du|de ce) (projet|livre) Gutenberg.*$",
    re.IGNORECASE | re.DOTALL,
)

# Marqueurs cherches uniquement dans la fin du texte (listes de corrections,
# achevés d'imprimer) : trop risque de les couper en plein milieu d'un livre.
TAIL_MARKERS = re.compile(
    r"[\s*\[]*(?:liste des (?:modifications|corrections)"
    r"|corrections?\s*:"
    r"|notes? du transcripteur\s*:"
    r"|notes? de transcription\s*:"
    r"|note du correcteur\s*:"
    r"|transcriber'?s?\s+notes?\s*:"
    r"|errata\b"
    r"|erreurs\s*:).*$",
    re.IGNORECASE | re.DOTALL,
)
IMPRINT = re.compile(
    r"[\s.,;–—-]*(?:\d{1,6}[\s.,;–—-]*)?"
    r"(?:imprimerie|imprimé? par|typographie|typ\.|imp\.|impr\.|impr\b"
    r"|lib(?:rairie)?\.?\s*-?\s*imp\w*)"
    r".{0,140}$",
    re.IGNORECASE,
)
# Code source embarque en fin de fichier (certains vieux Etexts joignent le
# programme de mise en forme apres le texte du livre).
CODE_BLOCK = re.compile(
    r"(?:/\*.{0,4000}?\*/\s*)?#(?:include|define)\b.*$",
    re.DOTALL,
)
# Mentions de production / numerisation. Elles collent souvent au titre sans
# ponctuation : on coupe donc tout ce qui precede (en tete) ou tout ce qui suit
# (en queue), quitte a perdre quelques mots du livre.
CREDITS = re.compile(
    r"(?:online\s+)?distributed\s+proofread(?:ing|ers)(?:\s+team)?(?:\s+at\s+\S+)?"
    r"|proofreading\s+team(?:\s+at\s+\S+)?"
    r"|project\s+gutenberg(?:\s+(?:literary\s+archive\s+foundation|ebooks?|etexts?))?"
    r"|(?:produced|prepared|scanned|proofread|transcribed)\s+by[^.;]{0,150}"
    r"|numéris\w*\s+par[^.;]{0,150}"
    r"|images?\s+(?:généreusement|generously)[^.;]{0,150}"
    r"|\bbiblioth[eè]que\s+nationale\s+de\s+France\b|\bgallica\b"
    r"|\binternet\s+archive\b|\barchive\.org\b"
    r"|the\s+online\s+books?\s+page",
    re.IGNORECASE,
)
CREDITS_HEAD_MAX = 1200  # zone ou une mention de production entraine la coupe
# Marqueurs sans ambiguite cherches dans une fenetre de queue plus large.
TAIL_WIDE_MARKERS = re.compile(
    r"[\s*\[]*(?:table\s+des\s+mati[eè]res|tables?\s+des\s+chapitres"
    r"|table\s+analytique|achev[ée]\s+d.imprimer).*$",
    re.IGNORECASE | re.DOTALL,
)
# Un titre de rubrique en capitales dans la zone finale = table / sommaire.
TAIL_HEADING = re.compile(r"[\s*\[]*\b(?:TABLE|TABLES|SOMMAIRE|SOMMAIRES|INDEX)\b.*$", re.DOTALL)
TAIL_WIDE = 0.10
TAIL_WIDE_MIN = 8000

# Fenetre de recherche des marqueurs ci-dessus : les 12 % finaux, plafonnes a
# 4000 caracteres (au-dela le risque de couper en plein texte l'emporte).
TAIL_WINDOW = 0.12
TAIL_MAX = 4000

# --- Coupure de tete ----------------------------------------------------------
PG_START = re.compile(
    r"^.*?\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^*]*\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
HEAD_NOTE_START = re.compile(
    r"^\s*\[?\s*(?:notes?\s+(?:de|sur|concernant|pour)\s+(?:la\s+|cette\s+)?transcription"
    r"|notes?\s+(?:du|au)\s+(?:transcripteur|correcteur|lecteur)"
    r"|notes?\s+de\s+l['’]éditeur"
    r"|avertissement\s+d[eu]\s+transcripteur"
    r"|au\s+lecteur\b"
    r"|transcriber'?s?\s+notes?"
    r"|notes?\s*:)",
    re.IGNORECASE,
)
# Une premiere phrase qui parle de typographie ou de numerisation n'appartient
# pas au livre : on enchaine alors la suppression des phrases de meme nature.
STRONG_NOTE = re.compile(
    r"transcri|numéris|typograph|orthograph|coquille|ponctuation|graphie d"
    r"|erreurs?[^.]{0,40}corrig|caractères?[^.]{0,30}(?:grec|accentu|spéciaux)"
    r"|version (?:html|unicode)|fichier (?:digital|électronique|numérique)"
    r"|(?:édition|version) (?:électronique|numérique)|fichier pdf|produit à partir d",
    re.IGNORECASE,
)
HEAD_RULED_NOTE = re.compile(r"^\s*[-=*_]{3,}\s*(.{0,1500})\s*[-=*_]{3,}\s*", re.DOTALL)
NOTE_WORDS = re.compile(
    r"transcri|typograph|orthograph|corrig|coquille|italique|gras|soulign|majuscul"
    r"|accent|ligature|caractère|numéris|ponctuation|astérisque|parenthès"
    r"|versions? (?:html|unicode|epub|originale)|livre électronique|original"
    r"|(?:édition|version)s? (?:électronique|numérique|html|papier)|errata|planches?"
    r"|fichier (?:digital|électronique)|\blettres?\b"
    r"|note du transcripteur|notes? de bas de page|translittération|\bgrec\b"
    r"|\bnotes?\b|fichier pdf|produit à partir|diacritique|abréviation|exposant"
    r"|accolade|\bpolices?\b|guillemet|\bsignes?\b|\bmots?\b|\bpages? \d"
    r"|modifi|wikisource|google books|\bimages?\b",
    re.IGNORECASE,
)
HEAD_MAX = 2500  # on ne rogne jamais plus que ca en tete
# Blocs entre crochets de l'epoque "Etext" (en-tetes legaux, jeu de caracteres...)
HEAD_BOILERPLATE = re.compile(
    r"\s*\[[^\]]{0,600}?(?:project gutenberg|e-?text|copyright \(c\)|character set"
    r"|produced by|domaine public)[^\]]{0,600}?\]\s*",
    re.IGNORECASE,
)

# --- Nettoyages en ligne ------------------------------------------------------
BRACKET_NOTE = re.compile(
    r"\[\s*(?:illustration|image|note du transcripteur|notes? de transcription"
    r"|transcriber'?s?\s+notes?|note de l['’]éditeur|blank page|page blanche)[^\]]{0,600}\]?",
    re.IGNORECASE,
)
FOOTNOTE_REF = re.compile(r"\[\d{1,4}\]")
PRODUCED_BY = re.compile(
    r"\b(?:this )?(?:e-?text|e-?book|file|document|transcription)\s+(?:was |has been )?"
    r"(?:produced|prepared|scanned|proofread)\s+by[^.]{0,140}\.?",
    re.IGNORECASE,
)
URL = re.compile(r"(?:https?://|ftp://|www\.)\S+|<[^\s>]+@[^\s>]+>|\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
SEPARATOR = re.compile(r"(?:[*=~#_]\s*){3,}|(?:-\s*){4,}|(?:<>\s*){3,}")
HTML_TAG = re.compile(
    r"</?(?:p|br|hr|i|b|em|strong|u|ul|ol|li|dl|dt|dd|div|span|a|img|pre|code|center"
    r"|font|small|big|sup|sub|blockquote|table|thead|tbody|tr|td|th|h[1-6]"
    r"|html|head|body|title|meta|link)(?:\s[^<>]{0,200})?/?>",
    re.IGNORECASE,
)
# --- Suppression des pages de titre -------------------------------------------
# Le vrai texte commence a la premiere longue suite de mots en minuscules ; tout
# ce qui precede (titre, auteur, editeur, table des matieres) saute.
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
FRONT_MAX = 6000  # on ne cherche le debut du texte que dans cette zone
PROSE_RUN = 40  # nombre de mots examines
PROSE_RATIO = 0.80  # part de mots en minuscule attendue dans du vrai texte

LEADING_JUNK = re.compile(r"^[\s,;:.\-–—*_/\\)\]}]+")
ENTITY = re.compile(r"&(?:[a-zA-Z]{2,8}|#\d{1,5}|#x[0-9a-fA-F]{1,4});")
SPACES = re.compile(r"\s+")

INLINE_RULES = [
    ("notes_crochets", BRACKET_NOTE),
    ("appels_notes", FOOTNOTE_REF),
    ("mentions_producteur", PRODUCED_BY),
    ("balises_html", HTML_TAG),
    ("separateurs", SEPARATOR),
]


def cut_tail(text, stats):
    text, n = PG_END.subn("", text, count=1)
    if n:
        stats["fin_gutenberg"] = stats.get("fin_gutenberg", 0) + 1

    text, n = CODE_BLOCK.subn("", text, count=1)
    if n:
        stats["code_source"] = stats.get("code_source", 0) + 1

    split = min(len(text) - TAIL_WIDE_MIN, int(len(text) * (1 - TAIL_WIDE)))
    head, tail = text[:max(split, 0)], text[max(split, 0):]
    credit = CREDITS.search(tail)
    if credit:
        stats["credits_queue"] = stats.get("credits_queue", 0) + 1
        tail = tail[:credit.start()]
    tail, n = TAIL_WIDE_MARKERS.subn("", tail, count=1)
    if not n:
        tail, n = TAIL_HEADING.subn("", tail, count=1)
    if n:
        stats["table_finale"] = stats.get("table_finale", 0) + 1
    text = head + tail

    split = max(len(text) - TAIL_MAX, int(len(text) * (1 - TAIL_WINDOW)))
    head, tail = text[:split], text[split:]
    tail, n = TAIL_MARKERS.subn("", tail, count=1)
    if n:
        stats["listes_corrections"] = stats.get("listes_corrections", 0) + 1
    tail, n = IMPRINT.subn("", tail, count=1)
    if n:
        stats["achevés_imprimer"] = stats.get("achevés_imprimer", 0) + 1
    return head + tail


def drop_note_sentences(text, stats):
    """Coupe les phrases de tete qui parlent de la transcription, pas du livre."""
    window = text[:HEAD_MAX]
    sentences = re.split(r"(?<=[.!?])\s+", window)
    titled = bool(HEAD_NOTE_START.match(text[:200]))
    if not (titled or STRONG_NOTE.search(sentences[0])):
        return text, False

    # On coupe jusqu'a la derniere phrase de nature "note", en tolerant une
    # phrase neutre au milieu (une note est souvent entrecoupee d'exemples).
    # Un en-tete explicite ("Note du transcripteur", "Au lecteur") suffit a
    # condamner sa premiere phrase, meme si elle ne dit rien de la typographie.
    cut = 0
    position = 0
    clean_run = 0
    for index, sentence in enumerate(sentences):
        position += len(sentence) + 1
        if NOTE_WORDS.search(sentence) or (titled and index == 0):
            cut = position
            clean_run = 0
        else:
            clean_run += 1
            if clean_run > 1:
                break
    if cut:
        stats["note_de_tete"] = stats.get("note_de_tete", 0) + 1
        return text[cut:], True
    return text, False


def cut_head(text, stats):
    text, n = PG_START.subn("", text, count=1)
    if n:
        stats["debut_gutenberg"] = stats.get("debut_gutenberg", 0) + 1

    # En-tetes legaux de l'epoque "Etext", entre crochets et empiles.
    while True:
        head = HEAD_BOILERPLATE.match(text)
        if not head or head.end() > HEAD_MAX:
            break
        stats["entete_etext"] = stats.get("entete_etext", 0) + 1
        text = text[head.end():]

    # Notes de transcription entre crochets (crochets imbriques possibles).
    while True:
        stripped = text.lstrip()
        if not (stripped.startswith("[") and HEAD_NOTE_START.match(stripped[:200])):
            break
        depth = 0
        end = None
        for i, char in enumerate(stripped[:HEAD_MAX]):
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            break
        stats["note_de_tete"] = stats.get("note_de_tete", 0) + 1
        text = stripped[end:]

    # Note encadree par une ligne de tirets ou de signes egal.
    ruled = HEAD_RULED_NOTE.match(text)
    if ruled and NOTE_WORDS.search(ruled.group(1)):
        stats["note_encadrée"] = stats.get("note_encadrée", 0) + 1
        text = text[ruled.end():]

    # Note en clair, sans delimiteur : on la reconnait a son vocabulaire.
    text, fired = drop_note_sentences(text, stats)
    # Si une note vient d'etre coupee, on n'accepte plus qu'une mention de
    # production collee au nouveau debut : au-dela ce serait du texte du livre.
    limit = 250 if fired else CREDITS_HEAD_MAX

    # Mentions de production restantes (souvent collees au titre, sans point) :
    # seulement si aucune note n'a deja ete traitee, sinon on couperait au
    # milieu d'une phrase de cette note.
    for _ in range(3):
        last = None
        for match in CREDITS.finditer(text[:limit]):
            last = match
        if not last:
            break
        stats["credits_tete"] = stats.get("credits_tete", 0) + 1
        text = LEADING_JUNK.sub("", text[last.end():])
        text, _ = drop_note_sentences(text, stats)

    return text


def cut_front_matter(text, stats):
    words = [(m.start(), m.group(0)) for m in WORD.finditer(text[:FRONT_MAX])]
    if len(words) < PROSE_RUN + 10:
        return text
    lower = [word[0].islower() for _, word in words]
    for i, (start, word) in enumerate(words[:len(words) - PROSE_RUN]):
        if lower[i] or word.isupper():
            continue  # on cherche un debut de phrase, pas un titre en capitales
        if any(w.isupper() and len(w) > 2 for _, w in words[i + 1:i + 6]):
            continue  # un titre en capitales suit encore
        if sum(lower[i:i + PROSE_RUN]) >= PROSE_RATIO * PROSE_RUN:
            if start:
                stats["pages_de_titre"] = stats.get("pages_de_titre", 0) + 1
            return text[start:]
    return text


def clean(text, stats, keep_titles=False):
    original = len(text)
    text, n = URL.subn(" ", text)
    if n:
        stats["urls"] = stats.get("urls", 0) + n
    text = cut_tail(text, stats)
    text = cut_head(text, stats)
    for name, pattern in INLINE_RULES:
        text, n = pattern.subn(" ", text)
        if n:
            stats[name] = stats.get(name, 0) + n
    text, n = ENTITY.subn(lambda m: html.unescape(m.group(0)), text)
    if n:
        stats["entites_html"] = stats.get("entites_html", 0) + n
    text = SPACES.sub(" ", text).strip()
    if not keep_titles:
        text = cut_front_matter(text, stats)
    stats["caracteres_supprimes"] = stats.get("caracteres_supprimes", 0) + original - len(text)
    stats["caracteres_lus"] = stats.get("caracteres_lus", 0) + original
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DATA_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="defaut : <input-dir>_clean")
    parser.add_argument("--in-place", action="store_true", help="ecrase les fichiers sources")
    parser.add_argument("--dry-run", action="store_true", help="n'ecrit rien, affiche le rapport")
    parser.add_argument("--limit", type=int, default=None, help="ne traite que N fichiers")
    parser.add_argument("--keep-titles", action="store_true",
                        help="conserve les pages de titre et tables initiales")
    parser.add_argument("--preview", type=int, default=0,
                        help="affiche N debuts/fins de fichiers nettoyes")
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.is_dir():
        sys.exit(f"repertoire introuvable : {input_dir}")

    output_dir = input_dir if args.in_place else (
        args.output_dir or input_dir.with_name(input_dir.name + "_clean")
    )
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.txt"))[: args.limit]
    stats = {}
    for i, path in enumerate(files):
        text = clean(path.read_text(encoding="utf-8"), stats, args.keep_titles)
        if i < args.preview:
            print(f"\n=== {path.name} ===\n{text[:300]}\n  [...]\n{text[-300:]}\n")
        if not args.dry_run:
            (output_dir / path.name).write_text(text, encoding="utf-8")

    lus = stats.pop("caracteres_lus", 0)
    supprimes = stats.pop("caracteres_supprimes", 0)
    print(f"\n{len(files)} fichiers traites"
          f"{' (dry-run)' if args.dry_run else f' -> {output_dir}'}")
    for name in sorted(stats):
        print(f"  {name:24} {stats[name]}")
    ratio = 100 * supprimes / lus if lus else 0
    print(f"  {'caracteres supprimes':24} {supprimes} ({ratio:.2f} % de {lus})")


if __name__ == "__main__":
    main()
