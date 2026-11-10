"""Graphical input: draw a triangulated toric diagram in the browser (``python -m topvertex --draw``).

``index.html`` is a self-contained page (SVG + vanilla JS, no external resources) that also works
when opened as a plain file (export only); :mod:`topvertex.draw.server` serves it on 127.0.0.1 and
answers its ``/api/compute`` and ``/api/validate`` requests with the Python code.
"""
