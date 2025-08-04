import json
import re
import warnings
from typing import List, Optional, Tuple

import pandas
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def normalize_unicode_text(text: str) -> str:
    """
    Normalize Unicode characters in text data.

    This function handles common Unicode issues including:
    - En dashes (–) → regular hyphens (-)
    - Em dashes (—) → double hyphens (--)
    - Smart quotes ('') → straight quotes ('')
    - Smart quotes ("") → straight quotes ("")
    - Other typographic characters

    Args:
        text (str): Input text with potential Unicode characters

    Returns:
        str: Text with normalized Unicode characters
    """
    if not isinstance(text, str):
        return text

    # Unicode character replacements
    replacements = {
        "\u2013": "-",  # en dash (–)
        "\u2014": "--",  # em dash (—)
        "\u2018": "'",  # left single quotation mark (')
        "\u2019": "'",  # right single quotation mark (')
        "\u201c": '"',  # left double quotation mark (")
        "\u201d": '"',  # right double quotation mark (")
        "\u201f": '"',  # double low-9 quotation mark („)
        "\u201a": ",",  # single low-9 quotation mark (‚)
        "\u201e": '"',  # double low-9 quotation mark („)
        "\u2039": "<",  # single left-pointing angle quotation mark (‹)
        "\u203a": ">",  # single right-pointing angle quotation mark (›)
        "\u2026": "...",  # horizontal ellipsis (…)
        "\u2010": "-",  # hyphen (‐)
        "\u00b0": " degrees",  # degree sign (°)
        "\u2022": "•",  # bullet (•)
        "\u00a0": " ",  # non-breaking space
        "\u200b": "",  # zero-width space
        "\u200c": "",  # zero-width non-joiner
        "\u200d": "",  # zero-width joiner
        "\u2060": "",  # word joiner
    }

    # Apply replacements
    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)

    # Clean up multiple spaces and normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def clean_json_text_data(data, text_fields=None):
    """
    Clean Unicode characters in JSON data, specifically targeting text fields.

    Args:
        data: JSON data (dict, list, or string)
        text_fields: List of field names that contain text to clean.
                    If None, will attempt to clean all string values.

    Returns:
        Cleaned data with normalized Unicode characters
    """
    if isinstance(data, str):
        return normalize_unicode_text(data)

    elif isinstance(data, list):
        return [clean_json_text_data(item, text_fields) for item in data]

    elif isinstance(data, dict):
        cleaned_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Clean if it's a text field or if no specific fields specified
                if text_fields is None or key in text_fields:
                    cleaned_data[key] = normalize_unicode_text(value)
                else:
                    cleaned_data[key] = value
            else:
                # Recursively clean nested structures
                cleaned_data[key] = clean_json_text_data(value, text_fields)
        return cleaned_data

    else:
        return data


def write_text(
    file_path: str, title_list: List[str], description_list: Optional[List[str]] = None
) -> None:
    with open(file_path, "w") as f:
        if description_list is None:
            for item in title_list:
                f.write(f"{item}\n")


def remove_text_in_brackets(input_string: str) -> str:
    bracket_pattern = r"\(.*?\)"
    result = re.sub(bracket_pattern, "", input_string)
    return result


def process_urls(text: str) -> Tuple[str, List[str]]:
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    urls = url_pattern.findall(text)
    cleaned_text = url_pattern.sub("", text)
    return cleaned_text, urls


def remove_substrings(text: str, substrings_to_remove: List[str]) -> str:
    for substr in substrings_to_remove:
        text = text.replace(substr, "")
    return text.strip()


def merge_title_and_description_strings(
    df: "pandas.DataFrame",
    title_col: str = "snippet.title",
    description_col: str = "snippet.description",
) -> List[str]:
    docs: List[str] = []
    titles = df[title_col].tolist()
    descriptions = df[description_col].tolist()

    for t, d in zip(titles, descriptions):
        if isinstance(d, str):
            # If an identical title string is duplicated in the description, remove it
            d = d.replace(t, "").strip()
            # and then merge them. Will not catch fuzzier matches, obviously.
            text = ". ".join([t.strip(), d.replace("\n", "").strip()])
        else:
            text = t.strip()
        docs.append(text.strip())
    return docs


def clean_html(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def parse_nested_json(text: str) -> str:
    """Identify and parse JSON-like structures within text."""
    try:
        # Attempt to find and parse JSON-like structures
        start_idx = text.find("{")
        if start_idx != -1:
            json_str = text[start_idx:]
            parsed_json = json.loads(json_str)
            # Extract the meaningful content from the parsed JSON
            text = text[:start_idx] + json.dumps(parsed_json, indent=2)
    except json.JSONDecodeError:
        # If parsing fails, return the original text
        pass
    return text


def clean_comment_text(text: str) -> str:
    """Clean the comment text by removing HTML and parsing nested JSON."""
    if isinstance(text, str):
        text = clean_html(text)
        text = parse_nested_json(text)
        return text
    return ""  # Return an empty string if the text is not a valid string
