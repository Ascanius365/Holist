import re


def extract_result(text, tag="tag"):
    pattern = rf"<{tag}>([\s\S]*?)<\/{tag}>"
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        return "", False
    else:
        return matches[0], True


def extract_yes_no(text):
    is_yes = False
    match = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
    if match:
        is_yes = match.group(0).lower() == "yes"
    return is_yes


def prefix_exchanges_with_idx(exchanges):
    """
    Add index prefixes to conversation exchanges.
    
    Args:
        exchanges (list): List of exchange strings
        
    Returns:
        str: Formatted string with indexes
    """
    exchanges_str_with_idx = ""
    for i, exchange in enumerate(exchanges):
        exchanges_str_with_idx += f"[Exchange {i}]: {exchange}\n\n"
    return exchanges_str_with_idx 