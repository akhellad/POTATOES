from label_studio_ml.api import init_app
from backend import PotatoDetector

app = init_app(model_class=PotatoDetector)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090, debug=False)