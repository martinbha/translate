"""Map Whisper language codes (ISO 639-1) to NLLB FLORES-200 codes."""

# Whisper emits ISO 639-1 codes; NLLB wants FLORES-200 "lang_Script" codes.
WHISPER_TO_NLLB: dict[str, str] = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "pl": "pol_Latn",
    "cs": "ces_Latn",
    "sk": "slk_Latn",
    "ro": "ron_Latn",
    "el": "ell_Grek",
    "tr": "tur_Latn",
    "ar": "arb_Arab",
    "he": "heb_Hebr",
    "fa": "pes_Arab",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ur": "urd_Arab",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh": "zho_Hans",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "sv": "swe_Latn",
    "no": "nob_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "hu": "hun_Latn",
    "bg": "bul_Cyrl",
    "sr": "srp_Cyrl",
    "hr": "hrv_Latn",
    "sl": "slv_Latn",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "et": "est_Latn",
    "ca": "cat_Latn",
    "gl": "glg_Latn",
    "eu": "eus_Latn",
    "is": "isl_Latn",
    "ga": "gle_Latn",
    "cy": "cym_Latn",
    "sw": "swh_Latn",
    "af": "afr_Latn",
    "tl": "tgl_Latn",
}

ENGLISH_NLLB = "eng_Latn"


def to_nllb(whisper_code: str) -> str | None:
    """Return the NLLB code, or None if unknown / already English."""
    if not whisper_code:
        return None
    return WHISPER_TO_NLLB.get(whisper_code.lower())
