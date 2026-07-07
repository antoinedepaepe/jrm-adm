from ast import literal_eval
from pathlib import Path


def _parse_value(value):
    if value in ("true", "false"):
        return value == "true"
    if value in ("null", "none"):
        return None
    try:
        return literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def load_yaml(path):
    data = {}
    stack = [(-1, data)]

    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue

        indent = len(raw) - len(raw.lstrip(" "))
        key, value = line.strip().split(":", 1)

        while indent <= stack[-1][0]:
            stack.pop()

        if value.strip():
            stack[-1][1][key] = _parse_value(value.strip())
        else:
            stack[-1][1][key] = {}
            stack.append((indent, stack[-1][1][key]))

    return data
