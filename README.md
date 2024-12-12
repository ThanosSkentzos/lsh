Easy install of dependencies using uv:
https://docs.astral.sh/uv/getting-started/installation/
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync # install dependencies from pyproject.toml
source .venv/bin/activate
```
Otherwise use requirements.txt
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run our code by running
```
python main.py 1234
```
To duplicate our 5 seeds results run
```bash
time ./run.sh #To run the experiment
```
