import sys
from dataclasses import dataclass, field
import bibtexparser

# TODO: create default lists of fields to delete (depending on type of bib entry ('inproceedings', 'article', etc.))

# Global parameters
FILENAME = '94_useful_little_scripts\\01_tex_related\\01_bibfile_editor\\testfile.bib'
FIELDSTODELETE = ['address', 'isbn']


def main():

    # Select options
    opts = BibEditOptions()
    opts.fieldsToDelete = FIELDSTODELETE

    processfile(FILENAME, opts)

    return None


@dataclass
class BibEditOptions:
    fieldsToDelete: list[str] = field(default_factory=list)
    exportBack: bool = True     # if True, export the modified .bib file
    encoding: str = "utf8"      # .bib file encoding (UTF-8 is Zotero's export default)


def processfile(filename, opts: BibEditOptions):
    """
    Processes bib file according to selected processing options.

    Parameters
    ----------
    filename : str
        Name of the .bib file (including (relative) path and extension).
    opts : BibEditOptions object
        Processing options.
    """

    # Check inputs
    if filename[-4:]!= '.bib':
        raise ValueError('The input `filename` must include the ".bib" extension.')

    # Read .bib file with bibtexparser
    with open(filename, encoding=opts.encoding) as bibtex_file:
        bib_database = bibtexparser.bparser.BibTexParser(common_strings=True).parse_file(bibtex_file)

    # Delete unwanted fields for each bib entry
    for fieldToDelete in opts.fieldsToDelete:
        for ii in range(len(bib_database.entries)):
            _ = bib_database.entries[ii].pop(fieldToDelete)

    if opts.exportBack:
        # Export back
        exportFilename = f'{filename[:-4]}_modified.bib'
        with open(exportFilename, 'w', encoding=opts.encoding) as bibtex_file:
            bibtexparser.dump(bib_database, bibtex_file)
            print(f'Modified library exported to "{exportFilename}".')

    return None


if __name__ == '__main__':
    sys.exit(main())