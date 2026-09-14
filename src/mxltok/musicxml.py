"""Read and write MusicXML (``.musicxml``, ``.xml`` and compressed ``.mxl``)."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
import zipfile
from fractions import Fraction
from pathlib import Path

from mxltok.errors import MxlTokError
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


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def read_musicxml(source: str | Path | bytes | ET.Element) -> Score:
    """Read a score from a MusicXML file, its bytes or a parsed root element.

    Raises:
        MxlTokError: If the file is not valid MusicXML.
    """
    try:
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
    except (ET.ParseError, zipfile.BadZipFile) as exc:
        raise MxlTokError(f"Cannot parse {source!r}: {exc}") from exc
    return _parse_score(root)


def _read_mxl(path: Path) -> bytes:
    """Return the MusicXML document inside a compressed ``.mxl`` file."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        inner = None
        if "META-INF/container.xml" in names:
            container = ET.fromstring(archive.read("META-INF/container.xml"))
            rootfile = container.find(".//rootfile")
            if rootfile is not None:
                inner = rootfile.get("full-path")
        if inner not in names:
            candidates = [
                name
                for name in names
                if not name.startswith("META-INF")
                and name.lower().endswith((".xml", ".musicxml"))
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
    score_parts: dict[str, ET.Element] = {}
    for entry in root.iterfind("part-list/score-part"):
        score_parts.setdefault(entry.get("id", ""), entry)
    parts = []
    for part_el in root.iterfind("part"):
        part = Part(id=part_el.get("id", ""))
        score_part = score_parts.get(part.id)
        if score_part is not None:
            part.name = score_part.findtext("part-name") or ""
            midi = score_part.find(".//midi-instrument")
            if midi is not None:
                part.midi_channel = _int(midi.findtext("midi-channel"))
                part.midi_program = _int(midi.findtext("midi-program"))
        part.measures = _parse_measures(part_el)
        parts.append(part)
    return Score(parts)


def _timewise_to_partwise(root: ET.Element) -> ET.Element:
    partwise = ET.Element("score-partwise", root.attrib)
    partwise.extend(child for child in root if child.tag != "measure")
    parts: dict[str, ET.Element] = {}
    for measure in root.iterfind("measure"):
        for part in measure.iterfind("part"):
            part_id = part.get("id", "")
            if part_id not in parts:
                parts[part_id] = ET.SubElement(partwise, "part", {"id": part_id})
            ET.SubElement(parts[part_id], "measure", measure.attrib).extend(part)
    return partwise


def _parse_measures(part_el: ET.Element) -> list[Measure]:
    divisions: int | None = None  # unknown until the first <divisions>
    measures = []
    for measure_el in part_el.iterfind("measure"):
        elements: list[Element] = []
        for el in measure_el:
            if el.tag == "attributes":
                divisions = _int(el.findtext("divisions")) or divisions
                attributes = _parse_attributes(el)
                if attributes is not None:
                    elements.append(attributes)
            elif el.tag == "note":
                elements.append(_parse_note(el, divisions))
            elif el.tag in ("backup", "forward"):
                # Without <divisions> a cursor move cannot be placed in time.
                duration = _duration(el, divisions)
                if duration is not None:
                    move = Backup if el.tag == "backup" else Forward
                    elements.append(move(duration))
            elif el.tag == "direction":
                elements.extend(_parse_direction(el))
            elif el.tag == "sound":
                elements.extend(_parse_sound(el))
            elif el.tag == "harmony":
                harmony = _parse_harmony(el)
                if harmony is not None:
                    elements.append(harmony)
            elif el.tag == "barline":
                barline = _parse_barline(el)
                if barline is not None:
                    elements.append(barline)
        measures.append(Measure(elements))
    return measures


def _parse_attributes(el: ET.Element) -> Attributes | None:
    beats, beat_type = el.find("time/beats"), el.find("time/beat-type")
    time = None
    if beats is not None and beat_type is not None:
        time = (
            _strict_int(beats.text, "beats"),
            _strict_int(beat_type.text, "beat-type"),
        )
    clefs = [
        Clef(
            sign=clef.findtext("sign") or None,
            line=_int(clef.findtext("line")),
            number=_strict_int(clef.get("number", "1"), "clef number"),
        )
        for clef in el.iterfind("clef")
    ]
    key = el.find("key/fifths")
    staves = el.find("staves")
    attributes = Attributes(
        key=None if key is None else _strict_int(key.text, "fifths"),
        time=time,
        staves=None if staves is None else _strict_int(staves.text, "staves"),
        clefs=clefs,
    )
    return None if attributes == Attributes() else attributes


def _duration(el: ET.Element, divisions: int | None) -> Fraction | None:
    text = el.findtext("duration")
    if text is None or not divisions:
        return None
    try:
        return Fraction(text.strip()) / divisions
    except ValueError as exc:
        raise MxlTokError(f"Invalid <duration>: {text!r}") from exc


def _parse_note(el: ET.Element, divisions: int | None) -> Note:
    note = Note(duration=_duration(el, divisions) or Fraction(0))
    grace = el.find("grace")
    if grace is not None:
        note.grace = True
        note.grace_slash = grace.get("slash") == "yes"
    note.chord = el.find("chord") is not None
    rest, pitch, unpitched = el.find("rest"), el.find("pitch"), el.find("unpitched")
    if rest is not None:
        note.rest = True
        note.measure_rest = rest.get("measure") == "yes"
    elif pitch is not None:
        note.step = pitch.findtext("step")
        note.octave = _int(pitch.findtext("octave"))
        note.alter = _float(pitch.findtext("alter"))
    else:
        note.unpitched = True
        if unpitched is not None:
            note.step = unpitched.findtext("display-step")
            note.octave = _int(unpitched.findtext("display-octave"))
    note.type_ = el.findtext("type")
    note.dots = len(el.findall("dot"))
    note.accidental = el.findtext("accidental") or None
    note.voice = _int(el.findtext("voice"), strict=True)
    note.staff = _int(el.findtext("staff"), strict=True)
    note.stem = el.findtext("stem")
    note.print_object = el.get("print-object") != "no"
    actual = el.findtext("time-modification/actual-notes")
    normal = el.findtext("time-modification/normal-notes")
    if actual and normal:
        note.time_modification = (
            _strict_int(actual, "actual-notes"),
            _strict_int(normal, "normal-notes"),
        )
    ties = [tie_type for tie in el.iterfind("tie") if (tie_type := tie.get("type"))]
    note.beams = tuple(beam.text for beam in el.iterfind("beam") if beam.text)
    notations = el.find("notations")
    if notations is not None:
        for tied in notations.iterfind("tied"):
            tied_type = tied.get("type")
            if tied_type and tied_type not in ties:
                ties.append(tied_type)
        _parse_notations(notations, note)
    note.ties = tuple(ties)
    return note


def _parse_notations(notations: ET.Element, note: Note) -> None:
    note.slurs = tuple(
        slur_type
        for slur in notations.iterfind("slur")
        if (slur_type := slur.get("type")) and slur_type != "continue"
    )
    note.tuplets = tuple(
        tuplet_type
        for tuplet in notations.iterfind("tuplet")
        if (tuplet_type := tuplet.get("type"))
    )
    articulations = notations.find("articulations")
    if articulations is not None:
        note.articulations = tuple(child.tag for child in articulations)
    ornaments = notations.find("ornaments")
    if ornaments is not None:
        note.ornaments = tuple(child.tag for child in ornaments)
        tremolo = ornaments.find("tremolo")
        if tremolo is not None:
            note.tremolo = (tremolo.get("type", "single"), tremolo.text or "3")
    note.fermata = notations.find("fermata") is not None
    note.arpeggiate = notations.find("arpeggiate") is not None
    note.glissandos = tuple(
        glissando_type
        for glissando in notations.iterfind("glissando")
        if (glissando_type := glissando.get("type"))
    )


def _parse_direction(el: ET.Element) -> list[Direction]:
    sound = el.find("sound")
    directions = _parse_sound(sound) if sound is not None else []
    for direction_type in el.iterfind("direction-type"):
        dynamics = direction_type.find("dynamics")
        if dynamics is not None:
            directions += [Direction("dynamics", child.tag) for child in dynamics]
        for kind in ("wedge", "pedal"):
            marking = direction_type.find(kind)
            if marking is not None and (value := marking.get("type")):
                directions.append(Direction(kind, value))
    return directions


def _parse_sound(el: ET.Element) -> list[Direction]:
    tempo = el.get("tempo")
    if tempo is None:
        return []
    try:
        return [Direction("tempo", str(round(float(tempo))))]
    except (ValueError, OverflowError) as exc:
        raise MxlTokError(f"Invalid tempo: {tempo!r}") from exc


def _parse_harmony(el: ET.Element) -> Harmony | None:
    root = el.findtext("root/root-step")
    if root is None:
        return None
    bass = el.findtext("bass/bass-step")
    return Harmony(
        root=root.strip(),
        root_alter=_harmony_alter(el.findtext("root/root-alter")),
        kind=(el.findtext("kind") or "").strip() or "major",
        bass=bass.strip() if bass is not None else None,
        bass_alter=_harmony_alter(el.findtext("bass/bass-alter")),
    )


def _harmony_alter(text: str | None) -> int:
    """Return a chord symbol alteration; non-integer alterations are dropped."""
    alter = _float(text)
    return int(alter) if alter.is_integer() else 0


def _parse_barline(el: ET.Element) -> Barline | None:
    repeat = el.find("repeat")
    barline = Barline(
        location=el.get("location", "right"),
        repeat=repeat.get("direction") if repeat is not None else None,
        style=el.findtext("bar-style") or None,
    )
    if barline.repeat is None and barline.style is None:
        return None
    return barline


def _strict_int(text: str | None, what: str) -> int:
    try:
        return int((text or "").strip())
    except ValueError as exc:
        raise MxlTokError(f"Invalid integer for <{what}>: {text!r}") from exc


def _int(text: str | None, strict: bool = False) -> int | None:
    """Parse an optional integer; invalid values raise only when ``strict``."""
    if text is None or not text.strip():
        return None
    try:
        return int(text.strip())
    except ValueError as exc:
        if strict:
            raise MxlTokError(f"Invalid integer: {text!r}") from exc
        return None


def _float(text: str | None) -> float:
    try:
        return float(text) if text else 0.0
    except ValueError:
        return 0.0


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def write_musicxml(score: Score, path: str | Path | None = None) -> str:
    """Serialize a score to MusicXML and return the document.

    Args:
        score: The score to write.
        path: Also write the document there; a ``.mxl`` path is compressed.
    """
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
    root.extend(_write_part(part) for part in score.parts)
    return root


def _write_part(part: Part) -> ET.Element:
    divisions = _divisions(part)
    part_el = ET.Element("part", {"id": part.id})
    open_slurs = 0
    for number, measure in enumerate(part.measures, start=1):
        measure_el = ET.SubElement(part_el, "measure", {"number": str(number)})
        for element in measure.elements:
            if isinstance(element, Attributes):
                measure_el.append(_write_attributes(element))
            elif isinstance(element, Note):
                note_el, open_slurs = _write_note(element, divisions, open_slurs)
                measure_el.append(note_el)
            elif isinstance(element, Backup | Forward):
                tag = "backup" if isinstance(element, Backup) else "forward"
                ET.SubElement(
                    ET.SubElement(measure_el, tag), "duration"
                ).text = _duration_text(element.duration, divisions)
            elif isinstance(element, Direction):
                measure_el.append(_write_direction(element))
            elif isinstance(element, Harmony):
                measure_el.append(_write_harmony(element))
            elif isinstance(element, Barline):
                measure_el.append(_write_barline(element))
    if len(part_el):
        _insert_divisions(part_el[0], divisions)
    return part_el


def _divisions(part: Part) -> int:
    """Return the smallest ``<divisions>`` expressing every duration exactly."""
    divisions = 1
    for measure in part.measures:
        for element in measure.elements:
            if isinstance(element, Note | Backup | Forward):
                divisions = math.lcm(divisions, element.duration.denominator)
    return divisions


def _duration_text(duration: Fraction, divisions: int) -> str:
    return str(int(duration * divisions))


def _insert_divisions(measure_el: ET.Element, divisions: int) -> None:
    """Add ``<divisions>`` to the head ``<attributes>`` of the first measure."""
    position = next(
        (
            i
            for i, child in enumerate(measure_el)
            if child.tag not in ("print", "barline")
        ),
        len(measure_el),
    )
    if position < len(measure_el) and measure_el[position].tag == "attributes":
        attributes_el = measure_el[position]
    else:
        attributes_el = ET.Element("attributes")
        measure_el.insert(position, attributes_el)
    divisions_el = ET.Element("divisions")
    divisions_el.text = str(divisions)
    attributes_el.insert(0, divisions_el)


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
        ET.SubElement(el, "grace", {"slash": "yes"} if note.grace_slash else {})
    if note.chord:
        ET.SubElement(el, "chord")
    if note.rest:
        ET.SubElement(el, "rest", {"measure": "yes"} if note.measure_rest else {})
    elif note.unpitched:
        unpitched = ET.SubElement(el, "unpitched")
        if note.step is not None:
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
    if note.time_modification is not None:
        time_mod = ET.SubElement(el, "time-modification")
        actual, normal = note.time_modification
        ET.SubElement(time_mod, "actual-notes").text = str(actual)
        ET.SubElement(time_mod, "normal-notes").text = str(normal)
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
    for tuplet in note.tuplets:
        ET.SubElement(el, "tuplet", {"type": tuplet})
    for slur in note.slurs:
        attributes = {"type": slur}
        if slur == "start":
            open_slurs += 1
            attributes["number"] = str(open_slurs)
        elif open_slurs > 0:
            attributes["number"] = str(open_slurs)
            open_slurs -= 1
        ET.SubElement(el, "slur", attributes)
    if note.fermata:
        ET.SubElement(el, "fermata")
    if note.articulations:
        group = ET.SubElement(el, "articulations")
        for articulation in note.articulations:
            ET.SubElement(group, articulation)
    if note.ornaments:
        group = ET.SubElement(el, "ornaments")
        for ornament in note.ornaments:
            if ornament == "tremolo" and note.tremolo is not None:
                tremolo_type, marks = note.tremolo
                ET.SubElement(group, "tremolo", {"type": tremolo_type}).text = marks
            else:
                ET.SubElement(group, ornament)
    if note.arpeggiate:
        ET.SubElement(el, "arpeggiate")
    for glissando in note.glissandos:
        ET.SubElement(el, "glissando", {"type": glissando})
    return el, open_slurs


def _write_direction(direction: Direction) -> ET.Element:
    el = ET.Element("direction")
    direction_type = ET.SubElement(el, "direction-type")
    if direction.kind == "dynamics":
        ET.SubElement(ET.SubElement(direction_type, "dynamics"), direction.value)
    elif direction.kind == "tempo":
        metronome = ET.SubElement(direction_type, "metronome")
        ET.SubElement(metronome, "beat-unit").text = "quarter"
        ET.SubElement(metronome, "per-minute").text = direction.value
        ET.SubElement(el, "sound", {"tempo": direction.value})
    else:  # wedge or pedal
        ET.SubElement(direction_type, direction.kind, {"type": direction.value})
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
