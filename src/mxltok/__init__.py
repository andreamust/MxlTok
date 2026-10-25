"""MxlTok: tokenize MusicXML scores into LMX token sequences, and back.

Example:
    >>> from mxltok import MXLTokenizer, write_musicxml
    >>> tokenizer = MXLTokenizer()
    >>> sequence = tokenizer.encode("score.mxl")
    >>> write_musicxml(tokenizer.decode(sequence.ids), "decoded.musicxml")
"""

from mxltok.bpe import BPE
from mxltok.categories import TokenCategory, classify_token
from mxltok.config import SpecialTokens, TokenizerConfig
from mxltok.errors import MxlTokError
from mxltok.musicxml import read_musicxml, write_musicxml
from mxltok.score import Score
from mxltok.tokenizer import MXLTokenizer, TokSequence
from mxltok.transpose import transpose

__version__ = "0.1.0"

__all__ = [
    "BPE",
    "MXLTokenizer",
    "MxlTokError",
    "Score",
    "SpecialTokens",
    "TokSequence",
    "TokenCategory",
    "TokenizerConfig",
    "__version__",
    "classify_token",
    "read_musicxml",
    "transpose",
    "write_musicxml",
]
