"""MusicXML (``.musicxml``/``.xml``/``.mxl``) reading and writing for :mod:`score`."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
import zipfile
from fractions import Fraction
from pathlib import Path

from mxltok.classes import MxlTokError
from mxltok.score import (
    Attributes,
    Backup,
    Barline,
    Clef,
    Direction,
    Element,
    Forward,
    Harmony,
    Measure,
    Note,
    Part,
    Score,
)

_ORNAMENT_TAGS = frozenset(
    {
        "trill-mark",
        "turn",
        "delayed-turn",
        "inverted-turn",
        "delayed-inverted-turn",
        "vertical-turn",
        "inverted-vertical-turn",
        "shake",
        "mordent",
        "inverted-mordent",
        "schleifer",
        "tremolo",
        "haydn",
    }
)
_XML_HEADER = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
    '"http://www.musicxml.org/dtds/partwise.dtd">\n'
)
_CONTAINER_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<container><rootfiles><rootfile full-path="score.xml" '
    'media-type="application/vnd.recordare.musicxml+xml"/></rootfiles></container>\n'
)


def load(source: str | Path | bytes | ET.Element) -> Score:
    """Parse a MusicXML file, its bytes, or an already-parsed root element."""
    if isinstance(source, ET.Element):
        root = source
    elif isinstance(source, bytes):
        root = ET.fromstring(source)
    else:
        path = Path(source)
        if path.suffix.lower() == ".mxl":
            root = ET.fromstring(_read_mxl(path))
        else:
            root = ET.parse(path).getroot()
    return _parse_score(root)


def _read_mxl(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        inner = None
        if "META-INF/container.xml" in names:
            container = ET.fromstring(archive.read("META-INF/container.xml"))
            rootfile = container.find(".//rootfile")
            if rootfile is not None:
                inner = rootfile.get("full-path")
        if inner is None or inner not in names:
            candidates = [
                n
                for n in names
                if not n.startswith("META-INF")
                and n.lower().endswith((".xml", ".musicxml"))
            ]
            if not candidates:
                raise MxlTokError(f"No MusicXML document found inside {path}")
            inner = candidates[0]
        return archive.read(inner)


def _parse_score(root: ET.Element) -> Score:
    if root.tag == "score-timewise":
        root = _timewise_to_partwise(root)
    if root.tag != "score-partwise":
        raise MxlTokError(f"Unsupported MusicXML root element <{root.tag}>")
    score_parts = {sp.get("id"): sp for sp in root.iterfind("part-list/score-part")}
    parts: list[Part] = []
    for part_el in root.iterfind("part"):
        part_id = part_el.get("id") or f"P{len(parts) + 1}"
        part = Part(id=part_id)
        score_part = score_parts.get(part_id)
        if score_part is not None:
            part.name = (score_part.findtext("part-name") or "").strip()
            part.midi_program = _int_text(
                score_part.find("midi-instrument/midi-program")
            )
            part.midi_channel = _int_text(
                score_part.find("midi-instrument/midi-channel")
            )
        part.measures = _parse_measures(part_el)
        parts.append(part)
    return Score(parts=parts)


def _timewise_to_partwise(root: ET.Element) -> ET.Element:
    partwise = ET.Element("score-partwise", root.attrib)
    for child in root:
        if child.tag != "measure":
            partwise.append(child)
    part_elements: dict[str, ET.Element] = {}
    for measure in root.iterfind("measure"):
        for part in measure.iterfind("part"):
            part_id = part.get("id", "")
            if part_id not in part_elements:
                part_elements[part_id] = ET.SubElement(
                    partwise, "part", {"id": part_id}
                )
            new_measure = ET.SubElement(
                part_elements[part_id], "measure", measure.attrib
            )
            new_measure.extend(list(part))
    return partwise


def _parse_measures(part_el: ET.Element) -> list[Measure]:
    divisions = 1
    measures: list[Measure] = []
    for measure_el in part_el.iterfind("measure"):
        elements: list[Element] = []
        for el in measure_el:
            match el.tag:
                case "attributes":
                    attributes, new_divisions = _parse_attributes(el)
                    divisions = new_divisions or divisions
                    if attributes is not None:
                        elements.append(attributes)
                case "note":
                    elements.append(_parse_note(el, divisions))
                case "backup":
                    elements.append(Backup(_parse_duration(el, divisions)))
                case "forward":
                    elements.append(Forward(_parse_duration(el, divisions)))
                case "direction":
                    elements.extend(_parse_direction(el))
                case "sound":
                    elements.extend(_parse_sound(el))
                case "harmony":
                    harmony = _parse_harmony(el)
                    if harmony is not None:
                        elements.append(harmony)
                case "barline":
                    barline = _parse_barline(el)
                    if barline is not None:
                        elements.append(barline)
        measures.append(Measure(elements))
    return measures


def _parse_attributes(el: ET.Element) -> tuple[Attributes | None, int | None]:
    divisions = _int_text(el.find("divisions"))
    time: tuple[int, int] | None = None
    beats, beat_type = el.findtext("time/beats"), el.findtext("time/beat-type")
    if beats is not None and beat_type is not None:
        time = (_int(beats, "time/beats"), _int(beat_type, "time/beat-type"))
    clefs = [
        Clef(
            sign=(clef.findtext("sign") or "G").strip(),
            line=_int_text(clef.find("line")),
            number=int(clef.get("number", "1")),
        )
        for clef in el.iterfind("clef")
    ]
    attributes = Attributes(
        key=_int_text(el.find("key/fifths")),
        time=time,
        staves=_int_text(el.find("staves")),
        clefs=clefs,
    )
    if attributes == Attributes():
        return None, divisions
    return attributes, divisions


def _parse_duration(el: ET.Element, divisions: int) -> Fraction:
    text = el.findtext("duration")
    if text is None:
        return Fraction(0)
    return Fraction(_int(text, "duration"), divisions)


def _parse_note(el: ET.Element, divisions: int) -> Note:
    note = Note(duration=_parse_duration(el, divisions))
    grace = el.find("grace")
    if grace is not None:
        note.grace = True
        note.grace_slash = grace.get("slash") == "yes"
    note.chord = el.find("chord") is not None
    rest = el.find("rest")
    pitch = el.find("pitch")
    unpitched = el.find("unpitched")
    if rest is not None:
        note.rest = True
        note.rest_measure = rest.get("measure") == "yes"
    elif pitch is not None:
        note.step = pitch.findtext("step")
        note.octave = _int_text(pitch.find("octave"))
        note.alter = float(pitch.findtext("alter") or 0)
    elif unpitched is not None:
        note.unpitched = True
        note.step = unpitched.findtext("display-step")
        note.octave = _int_text(unpitched.find("display-octave"))
    note.type_ = el.findtext("type")
    note.dots = len(el.findall("dot"))
    note.accidental = el.findtext("accidental")
    note.voice = _int_text(el.find("voice"))
    note.staff = _int_text(el.find("staff"))
    note.stem = el.findtext("stem")
    note.print_object = el.get("print-object") != "no"
    time_mod = el.find("time-modification")
    if time_mod is not None:
        note.time_mod = (
            _int(time_mod.findtext("actual-notes") or "1", "actual-notes"),
            _int(time_mod.findtext("normal-notes") or "1", "normal-notes"),
        )
    note.ties = tuple(t for tie in el.iterfind("tie") if (t := tie.get("type")))
    beams = sorted(el.iterfind("beam"), key=lambda b: int(b.get("number", "1")))
    note.beams = tuple((beam.text or "").strip() for beam in beams)
    notations = el.find("notations")
    if notations is not None:
        _parse_notations(notations, note)
    return note


def _parse_notations(notations: ET.Element, note: Note) -> None:
    if not note.ties:
        note.ties = tuple(
            t for tied in notations.iterfind("tied") if (t := tied.get("type"))
        )
    note.slurs = tuple(
        t
        for slur in notations.iterfind("slur")
        if (t := slur.get("type")) and t != "continue"
    )
    note.tuplets = tuple(
        t for tup in notations.iterfind("tuplet") if (t := tup.get("type"))
    )
    note.articulations = tuple(
        child.tag for group in notations.iterfind("articulations") for child in group
    )
    ornaments: list[str] = []
    for group in notations.iterfind("ornaments"):
        for child in group:
            if child.tag == "tremolo":
                ornaments.append(
                    f"tremolo:{child.get('type', 'single')}:{child.text or '3'}"
                )
            elif child.tag in _ORNAMENT_TAGS:
                ornaments.append(child.tag)
    note.ornaments = tuple(ornaments)
    extra: list[str] = []
    for child in notations:
        if child.tag in {"fermata", "arpeggiate"}:
            extra.append(child.tag)
        elif child.tag == "glissando" and child.get("type") in {"start", "stop"}:
            extra.append(f"glissando-{child.get('type')}")
    note.notations = tuple(extra)


def _parse_direction(el: ET.Element) -> list[Direction]:
    directions: list[Direction] = []
    for direction_type in el.iterfind("direction-type"):
        for dynamics in direction_type.iterfind("dynamics"):
            directions.extend(
                Direction("dynamics", child.tag)
                for child in dynamics
                if child.tag != "other-dynamics"
            )
        for wedge in direction_type.iterfind("wedge"):
            if wedge.get("type") in {"crescendo", "diminuendo", "stop"}:
                directions.append(Direction("wedge", wedge.get("type", "")))
        for pedal in direction_type.iterfind("pedal"):
            if pedal.get("type") in {"start", "stop", "change"}:
                directions.append(Direction("pedal", pedal.get("type", "")))
    for sound in el.iterfind("sound"):
        directions.extend(_parse_sound(sound))
    return directions


def _parse_sound(el: ET.Element) -> list[Direction]:
    tempo = el.get("tempo")
    if tempo is None:
        return []
    return [Direction("tempo", str(round(float(tempo))))]


def _parse_harmony(el: ET.Element) -> Harmony | None:
    root_step = el.findtext("root/root-step")
    if root_step is None:
        return None
    return Harmony(
        root=root_step.strip(),
        root_alter=round(float(el.findtext("root/root-alter") or 0)),
        kind=(el.findtext("kind") or "major").strip() or "major",
        bass=(el.findtext("bass/bass-step") or "").strip() or None,
        bass_alter=round(float(el.findtext("bass/bass-alter") or 0)),
    )


def _parse_barline(el: ET.Element) -> Barline | None:
    repeat = el.find("repeat")
    barline = Barline(
        location=el.get("location", "right"),
        repeat=repeat.get("direction") if repeat is not None else None,
        style=(el.findtext("bar-style") or "").strip() or None,
    )
    if barline.repeat is None and barline.style is None:
        return None
    return barline


def _int(text: str, what: str) -> int:
    try:
        return int(text.strip())
    except ValueError as exc:
        raise MxlTokError(f"Invalid integer for <{what}>: {text!r}") from exc


def _int_text(el: ET.Element | None) -> int | None:
    if el is None or el.text is None or not el.text.strip():
        return None
    return _int(el.text, el.tag)


def dump(score: Score, path: str | Path | None = None) -> str:
    """Serialize ``score`` to MusicXML. Writes ``path`` too (``.mxl`` = zipped)."""
    root = _write_score(score)
    ET.indent(root)
    text = _XML_HEADER + ET.tostring(root, encoding="unicode") + "\n"
    if path is not None:
        path = Path(path)
        if path.suffix.lower() == ".mxl":
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("META-INF/container.xml", _CONTAINER_XML)
                archive.writestr("score.xml", text)
        else:
            path.write_text(text, encoding="utf-8")
    return text


def _write_score(score: Score) -> ET.Element:
    root = ET.Element("score-partwise", {"version": "4.0"})
    part_list = ET.SubElement(root, "part-list")
    for part in score.parts:
        score_part = ET.SubElement(part_list, "score-part", {"id": part.id})
        ET.SubElement(score_part, "part-name").text = part.name
        if part.midi_program is not None or part.midi_channel is not None:
            instrument_id = f"{part.id}-I1"
            instrument = ET.SubElement(
                score_part, "score-instrument", {"id": instrument_id}
            )
            ET.SubElement(instrument, "instrument-name").text = part.name
            midi = ET.SubElement(score_part, "midi-instrument", {"id": instrument_id})
            if part.midi_channel is not None:
                ET.SubElement(midi, "midi-channel").text = str(part.midi_channel)
            if part.midi_program is not None:
                ET.SubElement(midi, "midi-program").text = str(part.midi_program)
    for part in score.parts:
        root.append(_write_part(part))
    return root


def _write_part(part: Part) -> ET.Element:
    divisions = _divisions(part)
    part_el = ET.Element("part", {"id": part.id})
    open_slurs = 0
    for number, measure in enumerate(part.measures, start=1):
        measure_el = ET.SubElement(part_el, "measure", {"number": str(number)})
        divisions_pending = number == 1
        for element in measure.elements:
            match element:
                case Attributes():
                    attributes_el = _write_attributes(element)
                    if divisions_pending:
                        _insert_divisions(attributes_el, divisions)
                        divisions_pending = False
                    measure_el.append(attributes_el)
                case Note():
                    if divisions_pending:
                        _insert_divisions(
                            ET.SubElement(measure_el, "attributes"), divisions
                        )
                        divisions_pending = False
                    note_el, open_slurs = _write_note(element, divisions, open_slurs)
                    measure_el.append(note_el)
                case Backup() | Forward():
                    if divisions_pending:
                        _insert_divisions(
                            ET.SubElement(measure_el, "attributes"), divisions
                        )
                        divisions_pending = False
                    tag = "backup" if isinstance(element, Backup) else "forward"
                    el = ET.SubElement(measure_el, tag)
                    ET.SubElement(el, "duration").text = _duration_text(
                        element.duration, divisions
                    )
                case Direction():
                    measure_el.append(_write_direction(element))
                case Harmony():
                    measure_el.append(_write_harmony(element))
                case Barline():
                    measure_el.append(_write_barline(element))
        if divisions_pending:
            _insert_divisions(ET.SubElement(measure_el, "attributes"), divisions)
    return part_el


def _divisions(part: Part) -> int:
    divisions = 1
    for measure in part.measures:
        for element in measure.elements:
            if isinstance(element, Note | Backup | Forward):
                divisions = math.lcm(divisions, element.duration.denominator)
    return divisions


def _duration_text(duration: Fraction, divisions: int) -> str:
    return str(int(duration * divisions))


def _insert_divisions(attributes_el: ET.Element, divisions: int) -> None:
    el = ET.Element("divisions")
    el.text = str(divisions)
    attributes_el.insert(0, el)


def _write_attributes(attributes: Attributes) -> ET.Element:
    el = ET.Element("attributes")
    if attributes.key is not None:
        ET.SubElement(ET.SubElement(el, "key"), "fifths").text = str(attributes.key)
    if attributes.time is not None:
        time_el = ET.SubElement(el, "time")
        ET.SubElement(time_el, "beats").text = str(attributes.time[0])
        ET.SubElement(time_el, "beat-type").text = str(attributes.time[1])
    if attributes.staves is not None:
        ET.SubElement(el, "staves").text = str(attributes.staves)
    for clef in attributes.clefs:
        clef_el = ET.SubElement(el, "clef", {"number": str(clef.number)})
        ET.SubElement(clef_el, "sign").text = clef.sign
        if clef.line is not None:
            ET.SubElement(clef_el, "line").text = str(clef.line)
    return el


def _write_note(note: Note, divisions: int, open_slurs: int) -> tuple[ET.Element, int]:
    el = ET.Element("note")
    if not note.print_object:
        el.set("print-object", "no")
    if note.grace:
        grace = ET.SubElement(el, "grace")
        if note.grace_slash:
            grace.set("slash", "yes")
    if note.chord:
        ET.SubElement(el, "chord")
    if note.rest:
        rest = ET.SubElement(el, "rest")
        if note.rest_measure:
            rest.set("measure", "yes")
    elif note.unpitched:
        unpitched = ET.SubElement(el, "unpitched")
        ET.SubElement(unpitched, "display-step").text = note.step
        ET.SubElement(unpitched, "display-octave").text = str(note.octave)
    else:
        pitch = ET.SubElement(el, "pitch")
        ET.SubElement(pitch, "step").text = note.step
        if note.alter:
            ET.SubElement(pitch, "alter").text = _number_text(note.alter)
        ET.SubElement(pitch, "octave").text = str(note.octave)
    if not note.grace:
        ET.SubElement(el, "duration").text = _duration_text(note.duration, divisions)
    for tie in note.ties:
        ET.SubElement(el, "tie", {"type": tie})
    if note.voice is not None:
        ET.SubElement(el, "voice").text = str(note.voice)
    if note.type_ is not None:
        ET.SubElement(el, "type").text = note.type_
    for _ in range(note.dots):
        ET.SubElement(el, "dot")
    if note.accidental is not None:
        ET.SubElement(el, "accidental").text = note.accidental
    if note.time_mod is not None:
        time_mod = ET.SubElement(el, "time-modification")
        ET.SubElement(time_mod, "actual-notes").text = str(note.time_mod[0])
        ET.SubElement(time_mod, "normal-notes").text = str(note.time_mod[1])
    if note.stem is not None:
        ET.SubElement(el, "stem").text = note.stem
    if note.staff is not None:
        ET.SubElement(el, "staff").text = str(note.staff)
    for number, beam in enumerate(note.beams, start=1):
        ET.SubElement(el, "beam", {"number": str(number)}).text = beam
    notations, open_slurs = _write_notations(note, open_slurs)
    if len(notations):
        el.append(notations)
    return el, open_slurs


def _write_notations(note: Note, open_slurs: int) -> tuple[ET.Element, int]:
    el = ET.Element("notations")
    for tie in note.ties:
        ET.SubElement(el, "tied", {"type": tie})
    for slur in note.slurs:
        if slur == "start":
            open_slurs += 1
            ET.SubElement(el, "slur", {"type": slur, "number": str(open_slurs)})
        else:
            ET.SubElement(el, "slur", {"type": slur, "number": str(max(open_slurs, 1))})
            open_slurs = max(open_slurs - 1, 0)
    for tuplet in note.tuplets:
        ET.SubElement(el, "tuplet", {"type": tuplet})
    for notation in note.notations:
        if notation.startswith("glissando-"):
            ET.SubElement(
                el, "glissando", {"type": notation.removeprefix("glissando-")}
            )
        else:
            ET.SubElement(el, notation)
    if note.articulations:
        group = ET.SubElement(el, "articulations")
        for articulation in note.articulations:
            ET.SubElement(group, articulation)
    if note.ornaments:
        group = ET.SubElement(el, "ornaments")
        for ornament in note.ornaments:
            if ornament.startswith("tremolo:"):
                _, kind, marks = ornament.split(":")
                ET.SubElement(group, "tremolo", {"type": kind}).text = marks
            else:
                ET.SubElement(group, ornament)
    return el, open_slurs


def _write_direction(direction: Direction) -> ET.Element:
    el = ET.Element("direction")
    direction_type = ET.SubElement(el, "direction-type")
    match direction.kind:
        case "dynamics":
            ET.SubElement(ET.SubElement(direction_type, "dynamics"), direction.value)
        case "wedge":
            ET.SubElement(direction_type, "wedge", {"type": direction.value})
        case "pedal":
            ET.SubElement(direction_type, "pedal", {"type": direction.value})
        case "tempo":
            metronome = ET.SubElement(direction_type, "metronome")
            ET.SubElement(metronome, "beat-unit").text = "quarter"
            ET.SubElement(metronome, "per-minute").text = direction.value
            ET.SubElement(el, "sound", {"tempo": direction.value})
    return el


def _write_harmony(harmony: Harmony) -> ET.Element:
    el = ET.Element("harmony")
    root = ET.SubElement(el, "root")
    ET.SubElement(root, "root-step").text = harmony.root
    if harmony.root_alter:
        ET.SubElement(root, "root-alter").text = str(harmony.root_alter)
    ET.SubElement(el, "kind").text = harmony.kind
    if harmony.bass is not None:
        bass = ET.SubElement(el, "bass")
        ET.SubElement(bass, "bass-step").text = harmony.bass
        if harmony.bass_alter:
            ET.SubElement(bass, "bass-alter").text = str(harmony.bass_alter)
    return el


def _write_barline(barline: Barline) -> ET.Element:
    el = ET.Element("barline", {"location": barline.location})
    if barline.style is not None:
        ET.SubElement(el, "bar-style").text = barline.style
    if barline.repeat is not None:
        ET.SubElement(el, "repeat", {"direction": barline.repeat})
    return el


def _number_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)
