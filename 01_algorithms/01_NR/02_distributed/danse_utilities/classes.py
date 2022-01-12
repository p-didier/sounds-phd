from dataclasses import dataclass, fields
import csv

@dataclass
class ProgramSettings:
    """Class for keeping track of global simulation settings"""
    samplingFrequency: int
    signalDuration: float

    @classmethod
    def load(cls, filename: str):
        """Loads program settings object from file"""
        csv_reader = list(csv.DictReader(open(filename, "r")))
        p = cls(samplingFrequency=csv_reader[0]['samplingFrequency']
                    , signalDuration=csv_reader[0]['signalDuration']
        )
        print(f'Program settings loaded from file: "{filename}".')
        return p
    
    def save(self, filename: str):
        """Saves program settings as CSV so they can be loaded again later"""
        with open(filename, "w") as f:
            fieldnames = [field.name for field in fields(self)]
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            rows = [{fn: str(getattr(self, fn)) for fn in fieldnames}]
            csv_writer.writeheader()
            csv_writer.writerows(rows)
        f.close()
        print(f'Program settings save to file: "{filename}".')


@dataclass
class Results:
    # """Class for keeping track of global simulation settings"""
    # data: 
    a = 1