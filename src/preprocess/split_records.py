from pathlib import Path


def list_mitdb_records(mitdb_dir):
    mitdb_dir = Path(mitdb_dir)
    recs = sorted({p.stem for p in mitdb_dir.glob('*.hea') if p.stem.isdigit()})
    return recs


def split_records(records):
    n = len(records)
    train_val_end = int(round(0.9 * n))
    train_val = records[:train_val_end]
    test = records[train_val_end:]
    train_end = int(round(0.8 * len(train_val)))
    train = train_val[:train_end]
    val = train_val[train_end:]
    return {'train': train, 'val': val, 'test': test}
