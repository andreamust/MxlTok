"""General MIDI instrument families and the part -> family resolution.

Every part of a score is labelled with one of the 19 :data:`GM_CATEGORIES`
(the 16 General MIDI program families plus ``Drums``, ``Voice`` and ``Other``),
which becomes its ``part:instrument:<category>`` token.
"""

from __future__ import annotations

import re
from difflib import get_close_matches

from mxltok.score import Part

GM_CATEGORIES: tuple[str, ...] = (
    "Piano",  # GM programs 1-8
    "Chromatic-Percussion",  # 9-16
    "Organ",  # 17-24
    "Guitar",  # 25-32
    "Bass",  # 33-40
    "Strings",  # 41-48
    "Ensemble",  # 49-56
    "Brass",  # 57-64
    "Reed",  # 65-72
    "Pipe",  # 73-80
    "Synth-Lead",  # 81-88
    "Synth-Pad",  # 89-96
    "Synth-Effects",  # 97-104
    "Ethnic",  # 105-112
    "Percussive",  # 113-120
    "Sound-Effects",  # 121-128
    "Drums",  # MIDI channel 10
    "Voice",  # vocal parts
    "Other",  # fallback
)

DRUMS_MIDI_CHANNEL = 10
DEFAULT_MIDI_CHANNEL = 1

# One representative General MIDI program per category, used when decoding.
CATEGORY_TO_MIDI_PROGRAM: dict[str, int] = {
    "Piano": 1,
    "Chromatic-Percussion": 9,
    "Organ": 17,
    "Guitar": 25,
    "Bass": 33,
    "Strings": 41,
    "Ensemble": 49,
    "Brass": 57,
    "Reed": 65,
    "Pipe": 73,
    "Synth-Lead": 81,
    "Synth-Pad": 89,
    "Synth-Effects": 97,
    "Ethnic": 105,
    "Percussive": 113,
    "Sound-Effects": 121,
    "Drums": 1,
    "Voice": 53,
    "Other": 1,
}

# Lower-case instrument names (in several languages) and their category.
# The order matters: the first name contained in a part name wins.
INSTRUMENT_NAME_TO_CATEGORY: dict[str, str] = {
    # Piano family
    "piano": "Piano",
    "pianoforte": "Piano",
    "keyboard": "Piano",
    "harpsichord": "Piano",
    "clavichord": "Piano",
    "celesta": "Piano",
    "celeste": "Piano",
    "fortepiano": "Piano",
    "klavier": "Piano",
    "cembalo": "Piano",
    "grand piano": "Piano",
    "electric piano": "Piano",
    "e-piano": "Piano",
    "pianos": "Piano",
    "pfte": "Piano",
    "pf": "Piano",
    # Chromatic Percussion
    "vibraphone": "Chromatic-Percussion",
    "marimba": "Chromatic-Percussion",
    "xylophone": "Chromatic-Percussion",
    "glockenspiel": "Chromatic-Percussion",
    "tubular bells": "Chromatic-Percussion",
    "chimes": "Chromatic-Percussion",
    "bells": "Chromatic-Percussion",
    "crotales": "Chromatic-Percussion",
    "steel drums": "Chromatic-Percussion",
    "music box": "Chromatic-Percussion",
    # Organ
    "organ": "Organ",
    "harmonium": "Organ",
    "accordion": "Organ",
    "bandoneon": "Organ",
    "reed organ": "Organ",
    "pipe organ": "Organ",
    "church organ": "Organ",
    "concertina": "Organ",
    "harmonica": "Organ",
    "melodica": "Organ",
    # Guitar
    "guitar": "Guitar",
    "acoustic guitar": "Guitar",
    "electric guitar": "Guitar",
    "classical guitar": "Guitar",
    "nylon guitar": "Guitar",
    "steel guitar": "Guitar",
    "ukulele": "Guitar",
    "banjo": "Guitar",
    "mandolin": "Guitar",
    "mandola": "Guitar",
    "lute": "Guitar",
    "theorbo": "Guitar",
    "harp": "Guitar",
    "arpa": "Guitar",
    "zither": "Guitar",
    "balalaika": "Guitar",
    "bouzouki": "Guitar",
    "charango": "Guitar",
    "guitarra": "Guitar",
    "chitarra": "Guitar",
    "gitarre": "Guitar",
    # Bass
    "bass": "Bass",
    "electric bass": "Bass",
    "bass guitar": "Bass",
    "contrabass": "Bass",
    "double bass": "Bass",
    "string bass": "Bass",
    "upright bass": "Bass",
    "contrabasso": "Bass",
    "contrabbasso": "Bass",
    "kontrabass": "Bass",
    "basso continuo": "Bass",
    "continuo": "Bass",
    "basso": "Bass",
    # Strings
    "violin": "Strings",
    "violino": "Strings",
    "violine": "Strings",
    "viola": "Strings",
    "cello": "Strings",
    "violoncello": "Strings",
    "violoncelle": "Strings",
    "fiddle": "Strings",
    "strings": "Strings",
    "string ensemble": "Strings",
    "violins": "Strings",
    "violas": "Strings",
    "cellos": "Strings",
    "violini": "Strings",
    "vln": "Strings",
    "vla": "Strings",
    "vlc": "Strings",
    "vc": "Strings",
    # Ensemble
    "orchestra": "Ensemble",
    "ensemble": "Ensemble",
    "tutti": "Ensemble",
    "ripieno": "Ensemble",
    # Brass
    "trumpet": "Brass",
    "trombone": "Brass",
    "tuba": "Brass",
    "horn": "Brass",
    "french horn": "Brass",
    "cornet": "Brass",
    "euphonium": "Brass",
    "flugelhorn": "Brass",
    "bugle": "Brass",
    "sousaphone": "Brass",
    "tromba": "Brass",
    "corno": "Brass",
    "corni": "Brass",
    "trombe": "Brass",
    "tromboni": "Brass",
    "posaune": "Brass",
    "trompete": "Brass",
    "trumpets": "Brass",
    "trombones": "Brass",
    "horns": "Brass",
    "tubas": "Brass",
    "cornets": "Brass",
    "cor": "Brass",
    "cors": "Brass",
    "trp": "Brass",
    # Reed
    "clarinet": "Reed",
    "oboe": "Reed",
    "saxophone": "Reed",
    "bassoon": "Reed",
    "english horn": "Reed",
    "cor anglais": "Reed",
    "bass clarinet": "Reed",
    "contrabassoon": "Reed",
    "contrafagotto": "Reed",
    "oboe d'amore": "Reed",
    "heckelphone": "Reed",
    "alto saxophone": "Reed",
    "tenor saxophone": "Reed",
    "soprano saxophone": "Reed",
    "baritone saxophone": "Reed",
    "alto sax": "Reed",
    "tenor sax": "Reed",
    "soprano sax": "Reed",
    "bari sax": "Reed",
    "sax": "Reed",
    "clarinets": "Reed",
    "oboes": "Reed",
    "bassoons": "Reed",
    "fagotto": "Reed",
    "fagotti": "Reed",
    "clarinette": "Reed",
    "clarinetti": "Reed",
    "klarinette": "Reed",
    "hautbois": "Reed",
    "oboi": "Reed",
    "fagot": "Reed",
    "cl": "Reed",
    "ob": "Reed",
    "fg": "Reed",
    "bn": "Reed",
    # Pipe
    "flute": "Pipe",
    "piccolo": "Pipe",
    "recorder": "Pipe",
    "pan flute": "Pipe",
    "fife": "Pipe",
    "shakuhachi": "Pipe",
    "whistle": "Pipe",
    "ocarina": "Pipe",
    "alto flute": "Pipe",
    "bass flute": "Pipe",
    "flauto": "Pipe",
    "flauti": "Pipe",
    "flutes": "Pipe",
    "piccolos": "Pipe",
    "traverso": "Pipe",
    "flauto traverso": "Pipe",
    "blockflute": "Pipe",
    "fl": "Pipe",
    "picc": "Pipe",
    # Synth-Lead
    "synth lead": "Synth-Lead",
    "lead synth": "Synth-Lead",
    # Synth-Pad
    "synth pad": "Synth-Pad",
    "pad": "Synth-Pad",
    "synth": "Synth-Pad",
    "synthesizer": "Synth-Pad",
    # Synth-Effects
    "synth effects": "Synth-Effects",
    "fx": "Synth-Effects",
    # Ethnic
    "sitar": "Ethnic",
    "shamisen": "Ethnic",
    "koto": "Ethnic",
    "erhu": "Ethnic",
    "bagpipe": "Ethnic",
    "bagpipes": "Ethnic",
    "didgeridoo": "Ethnic",
    "kalimba": "Ethnic",
    "dulcimer": "Ethnic",
    "oud": "Ethnic",
    # Percussive
    "agogo": "Percussive",
    "woodblock": "Percussive",
    "taiko": "Percussive",
    # Sound-Effects
    "sound effects": "Sound-Effects",
    # Drums / Percussion
    "drums": "Drums",
    "drum": "Drums",
    "percussion": "Drums",
    "timpani": "Drums",
    "timpano": "Drums",
    "pauken": "Drums",
    "timbales": "Drums",
    "snare": "Drums",
    "snare drum": "Drums",
    "bass drum": "Drums",
    "cymbals": "Drums",
    "cymbal": "Drums",
    "triangle": "Drums",
    "tambourine": "Drums",
    "bongos": "Drums",
    "congas": "Drums",
    "conga": "Drums",
    "drum set": "Drums",
    "drum kit": "Drums",
    "drumset": "Drums",
    "kit": "Drums",
    "hi-hat": "Drums",
    "tom": "Drums",
    "tom-tom": "Drums",
    "castanets": "Drums",
    "claves": "Drums",
    "cowbell": "Drums",
    "guiro": "Drums",
    "maracas": "Drums",
    "shaker": "Drums",
    "suspended cymbal": "Drums",
    "crash cymbal": "Drums",
    "ride cymbal": "Drums",
    "gong": "Drums",
    "tam-tam": "Drums",
    "wood block": "Drums",
    "temple blocks": "Drums",
    "unpitched percussion": "Drums",
    "perc": "Drums",
    # Voice
    "voice": "Voice",
    "vocal": "Voice",
    "vocals": "Voice",
    "soprano": "Voice",
    "alto": "Voice",
    "tenor": "Voice",
    "baritone": "Voice",
    "mezzo-soprano": "Voice",
    "mezzo soprano": "Voice",
    "countertenor": "Voice",
    "singer": "Voice",
    "choir": "Voice",
    "chorus": "Voice",
    "chor": "Voice",
    "coro": "Voice",
    "sopran": "Voice",
    "soprani": "Voice",
    "contralto": "Voice",
    "bariton": "Voice",
    "tenore": "Voice",
    "tenori": "Voice",
    "bassi": "Voice",
    "singstimme": "Voice",
    "gesang": "Voice",
    "voce": "Voice",
    "voci": "Voice",
    "chant": "Voice",
    "soloist": "Voice",
    "solo": "Voice",
}


def instrument_category(part: Part) -> str:
    """Return the General MIDI category of a part.

    The part's MIDI metadata is used when present: channel 10 means
    ``Drums`` and a program number selects its family. Otherwise the part
    name is matched against :data:`INSTRUMENT_NAME_TO_CATEGORY`, exactly,
    then as a substring, then fuzzily. Unresolvable parts are ``Other``.
    """
    if part.midi_channel == DRUMS_MIDI_CHANNEL:
        return "Drums"
    if part.midi_program is not None and 1 <= part.midi_program <= 128:
        return GM_CATEGORIES[(part.midi_program - 1) // 8]

    name = _normalize_name(part.name or part.id)
    if not name:
        return "Other"
    if name in INSTRUMENT_NAME_TO_CATEGORY:
        return INSTRUMENT_NAME_TO_CATEGORY[name]
    for known, category in INSTRUMENT_NAME_TO_CATEGORY.items():
        if len(known) >= 3 and known in name:
            return category
    matches = get_close_matches(name, INSTRUMENT_NAME_TO_CATEGORY, n=1, cutoff=0.6)
    return INSTRUMENT_NAME_TO_CATEGORY[matches[0]] if matches else "Other"


def _normalize_name(name: str) -> str:
    """Lower-case a part name, keeping only letters, digits and ``-'./()``."""
    name = re.sub(r"[\t\n\r]+", " ", name)
    name = re.sub(r"[^a-zA-Z0-9\s\-'./()]", "", name)
    return re.sub(r"\s+", " ", name).strip().lower()
