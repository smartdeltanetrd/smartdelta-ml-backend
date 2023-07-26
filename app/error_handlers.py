from flask import Blueprint, jsonify, Response
from flask_json_schema import JsonValidationError

bp = Blueprint('error_handlers', __name__)

@bp.errorhandler(400)
def bad_request(error):
    return jsonify(message="Bad request", error=str(error)), 400

@bp.errorhandler(404)
def not_found(error):
    return jsonify(message="Not found", error=str(error)), 404

@bp.errorhandler(500)
def internal_server_error(error):
    return jsonify(message="Internal server error", error=str(error)), 500

@bp.errorhandler(JsonValidationError)
def validation_error(e: JsonValidationError) -> Response:
    return jsonify(message="JSON validation error", errors=e.message), 400
