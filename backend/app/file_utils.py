def _parse_range_header(range_header: str, file_size: int):
    try:
        units, _, rng = range_header.partition("=")
        if units.strip().lower() != "bytes":
            return None
        start_s, _, end_s = rng.partition("-")
        if start_s == "":
            return None
        start = int(start_s)
        end = int(end_s) if end_s else file_size - 1
        if start < 0 or end < start or end >= file_size:
            return None
        return start, end
    except Exception:
        return None