from datetime import datetime


def datetime_to_str(dt, use_short_weekday=False):
    """
    Convert a datetime object to a formatted string.

    Args:
        dt (datetime): Datetime object to convert
        use_short_weekday (bool): If True, use abbreviated weekday name (%a) instead of full name (%A)

    Returns:
        str: Formatted date string
    """
    weekday_format = "%a" if use_short_weekday else "%A"
    return dt.strftime(f"%Y-%m-%d {weekday_format} %H:%M:%S")


def str_to_datetime(date_str):
    """
    Parse a date string into a datetime object, supporting multiple formats.

    Args:
        date_str (str): Date string in one of the supported formats:
                       - "YYYY-MM-DD Weekday HH:MM:SS" (full or abbreviated weekday)
                       - "YYYY-MM-DD Weekday HH:MM" (full or abbreviated weekday)
                       - "YYYY-MM-DD HH:MM:SS" (no weekday)
                       - "YYYY-MM-DD HH:MM" (no weekday)

    Returns:
        datetime: Parsed datetime object

    Raises:
        ValueError: If the date string doesn't match any supported format
    """
    # Define all formats to try, in order of preference
    formats = [
        "%Y-%m-%d %A %H:%M:%S",  # Full weekday with seconds
        "%Y-%m-%d %a %H:%M:%S",  # Abbreviated weekday with seconds
        "%Y-%m-%d %A %H:%M",  # Full weekday without seconds
        "%Y-%m-%d %a %H:%M",  # Abbreviated weekday without seconds
        "%Y-%m-%d %H:%M:%S",  # No weekday with seconds
        "%Y-%m-%d %H:%M",  # No weekday without seconds
        "%Y/%m/%d (%A) %H:%M:%S",
        "%Y/%m/%d (%a) %H:%M:%S",
        "%Y/%m/%d (%A) %H:%M",
        "%Y/%m/%d (%a) %H:%M",
        "%I:%M %p on %d %B, %Y",  # e.g. "1:56 pm on 8 May, 2023"
        "%I:%M %p on %d %b, %Y",  # with abbreviated month name
    ]

    # Try each format
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # If none of the formats matched
    formats_str = ", ".join(f"'{fmt}'" for fmt in formats)
    raise ValueError(f"Invalid date format. Expected one of: {formats_str}, got '{date_str}'")
