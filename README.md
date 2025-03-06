Examples:

```bash
pip install mike
git checkout v1.0
python generate_mkdocs.py
mike deploy v1.0 --push
git checkout v1.1
python generate_mkdocs.py
mike deploy v1.1 latest --push
```