__all__ = ["create_app"]


def create_app():
    from .api import create_app as _create_app

    return _create_app()
