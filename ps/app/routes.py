from app import app

@app.route('/')
@app.route('/index')
def index():
    return "OIS.PyServices"

@app.route('/health/live')
def health():
    return 'live'