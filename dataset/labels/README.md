# Labels

Each `.txt` file uses YOLO bounding-box format:

```text
0 center_x center_y width height
```

Run the annotator from the project root:

```bash
python scripts/annotate.py --images dataset/images --labels dataset/labels
```
