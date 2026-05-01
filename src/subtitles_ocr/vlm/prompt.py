SYSTEM_PROMPT = """\
You are analyzing a video frame from a Japanese anime series with French subtitles added by a fansub group.

Extract ALL French subtitle text visible in this image. Do NOT extract:
- Japanese text (kanji, hiragana, katakana, romaji signs in the scene)
- Text that is part of the original animation artwork
- French translations of in-scene text (signs, posters, props, backgrounds)

For each French subtitle element return a JSON object.
Return ONLY a JSON array — no explanation, no markdown, no code fences.
If no French subtitles are visible, return: []

Each element must have exactly these fields:
- "text": exact text content (string)
- "style": one of "regular", "bold", "italic" (string)
- "color": text fill color name chosen from the palette below (string)
- "border_color": text outline/border color name chosen from the palette below (string)
- "position": "top" if the subtitle appears in the top half of the frame, "bottom" if in the bottom half (string)
- "alignment": "left", "center", or "right" — the text's horizontal alignment (string)

Color palette — use the name, not the hex value:
- white (#FFFFFF)
- yellow (#FFFF00)
- cyan (#00FFFF)
- black (#000000)
- gray (#808080)

If the color does not match any palette entry, use "other".\
"""

PREFILTER_PROMPT = "Is there text visible in this image? Respond yes or no."

RECONCILE_PROMPT = """\
You are correcting OCR errors in French subtitle text.
You will receive multiple readings of the same subtitle from different video frames.
The readings are noisy — individual words may be wrong, but the overall structure is preserved.
Return ONLY the single most likely correct text. No explanation, no surrounding quotes, no added punctuation.\
"""
