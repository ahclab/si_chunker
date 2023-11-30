## Installation
```
pip install requirements.txt
python -m spacy download en_core_web_md
python -c "import benepar; benepar.download('benepar_en3')"
```

## Chunk files
```
python chunker.py \
  --sentence-files sample_data/talk1.txt sample_data/talk2.txt \
  --output-dir sample_data/auto-chunks \
  --print \
  --disable-tqdm \
  --min-chunk 3
```
You can also specify the output file for each input file instead of `--output-dir` with `--output-files`.

## Evaluation
```
python chunker.py \
  --sentence-files sample_data/talk1.txt sample_data/talk2.txt \
  --do-eval \
  --ref-files sample_data/talk1_chunk.txt sample_data/talk2_chunk.txt \
  --output-dir sample_data/auto-chunks \
  --min-chunk 3

# F1 = 0.933, Precision = 0.875, Recall = 1.000
```

## Interactive Mode
```
python chunker.py --interactive
```