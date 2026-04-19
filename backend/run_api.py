from app.api import create_app


app = create_app()


if __name__ == "__main__":
    app.run(host=app.config["APP_CONFIG"].api_host, port=app.config["APP_CONFIG"].api_port, debug=True)

