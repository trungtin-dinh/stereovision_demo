from __future__ import annotations

import functools
import inspect

import streamlit as st


def _make_streamlit_width_compatible(function):
    """
    Keep app_sl.py unchanged while making old/new Streamlit width arguments
    compatible.

    Some versions support use_container_width, newer versions prefer
    width="stretch", and older versions support neither on some widgets.
    """
    try:
        signature = inspect.signature(function)
        parameters = signature.parameters
        supports_use_container_width = "use_container_width" in parameters
        supports_width = "width" in parameters
    except (TypeError, ValueError):
        supports_use_container_width = False
        supports_width = False

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if "use_container_width" in kwargs and not supports_use_container_width:
            use_container_width = bool(kwargs.pop("use_container_width"))
            if supports_width and use_container_width and "width" not in kwargs:
                kwargs["width"] = "stretch"

        if "width" in kwargs and not supports_width:
            kwargs.pop("width", None)

        return function(*args, **kwargs)

    return wrapper


def _patch_streamlit_width_arguments() -> None:
    for name in [
        "image",
        "plotly_chart",
        "button",
        "download_button",
        "dataframe",
    ]:
        if hasattr(st, name):
            setattr(st, name, _make_streamlit_width_compatible(getattr(st, name)))


def main() -> None:
    _patch_streamlit_width_arguments()

    import app_sl

    app_sl.main()


if __name__ == "__main__":
    main()
