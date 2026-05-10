from app.api import create_app


app = create_app()


if __name__ == "__main__":
    config = app.config["APP_CONFIG"]
    app.run(host=config.api_host, port=config.api_port, debug=config.debug, use_reloader=False)
