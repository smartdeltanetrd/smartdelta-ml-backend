from flask import Flask
from app import data_utils, error_handlers, routes, MessageList, LogAnalyzer

def create_app():
    app = Flask(__name__)
    app.config.from_object('app.constants')

    # Register the blueprints for routes and error handlers
    app.register_blueprint(data_utils.bp)
    app.register_blueprint(MessageList.bp)
    app.register_blueprint(LogAnalyzer.bp)
    app.register_blueprint(error_handlers.bp)
    app.register_blueprint(routes.bp)

    return app
