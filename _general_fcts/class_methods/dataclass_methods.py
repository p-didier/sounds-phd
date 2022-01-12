import csv
from dataclasses import fields

# @classmethod
def load(cls, filename: str):
    """Loads program settings object from file"""
    csv_reader = csv.DictReader(open(filename, "r"))
    csv_reader_list = list(csv_reader)
    p = cls()
    for field in csv_reader.fieldnames:
        val = csv_reader_list[0][field]
        try:
            val = float(val)
        except ValueError:
            val = val
        setattr(p, field, val)
    print(f'Program settings loaded from file: "{filename}".')
    return p

def save(self, filename: str):
    """Saves program settings as CSV so they can be loaded again later"""
    with open(filename, "w") as f:
        fieldnames = [field.name for field in fields(self)]
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        rows = [{fn: getattr(self, fn) for fn in fieldnames}]
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    f.close()
    print(f'Program settings saved to file: "{filename}".')