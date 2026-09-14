"""The single exception type raised by MxlTok."""


class MxlTokError(ValueError):
    """Raised for unreadable MusicXML, unencodable content or invalid tokens."""
