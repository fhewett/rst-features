VENV='rst-features'

echo "Preparing Python environment..."
python3 -m venv $VENV
source $VENV/bin/activate
pip install -r requirements.txt

echo "Running models..."
cd run_models
python run_models.py
