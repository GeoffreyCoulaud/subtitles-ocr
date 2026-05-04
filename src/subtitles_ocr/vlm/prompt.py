SYSTEM_PROMPT = """\
You are analyzing a video frame from a Japanese anime series with French subtitles added by a fansub group.

Extract ALL French subtitle text visible in this image. Do NOT extract:
- Japanese text (kanji, hiragana, katakana, romaji signs in the scene)
- Text that is part of the original animation artwork
- French translations of in-scene text (signs, posters, props, backgrounds)

Return ONLY a raw JSON object — no markdown, no code fences, no explanation. Start your response with { and end with }. Use this exact format:
{"subtitles": [{"text": "...", "style": "...", "color": "...", "position": "..."}, ...]}
If no French subtitles are visible, return: {"subtitles": []}

Each element in the "subtitles" array must be a JSON object with these fields:
- "text": exact text content (string) — REQUIRED
- "style": one of "regular", "italic" (string) — default: "regular"
- "color": text fill color name chosen from the palette below (string) — default: "white"
- "position": "top" if the subtitle appears in the top half of the frame, "bottom" if in the bottom half (string) — default: "bottom"

Color palette — use the name, not the hex value:
- white (#FFFFFF)
- yellow (#FFFF00)
- cyan (#00FFFF)

If the color does not match any palette entry, use "other".

Never include the same subtitle text more than once in the array.\
"""

PREFILTER_PROMPT = 'Is there text visible in this image? Return only a JSON object: {"has_text": true} or {"has_text": false}.'

RECONCILE_PROMPT = """\
You are correcting OCR errors in French subtitle text.
You will receive multiple readings of the same subtitle from different video frames.
The readings are noisy — individual words may be wrong, but the overall structure is preserved.
Return ONLY the single most likely correct text. No explanation, no surrounding quotes, no added punctuation.\
"""
