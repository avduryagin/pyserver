from app import app
from waitress import serve
import logging
import os

logger = logging.getLogger(__name__)

try:
    port = os.environ["PORT"]
except:
    port = 5000


def run():
    logger.info("App start")
    serve(app, host="0.0.0.0", port=port, threads=4)  # WAITRESS!

if __name__ == '__main__':
    run()
