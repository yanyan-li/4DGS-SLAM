import rich

_log_styles = {
    "4DGS-SLAM": "bold green",
    "Frontend": "bold yellow",
    "Backend": "bold red",
    "Optimizer": "bold blue",
    "Static": "bold blue",
    "Dynamic": "bold cyan",
    "GUI": "bold magenta",
    "Eval": "bold red",
}


def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="4DGS-SLAM"):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)
