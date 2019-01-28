import os
import pandas as pd

def read_rrf_file(filename, data_dir='../data', col_names=None):
    """
    Opens any RRF file in the UMLS Metathesuarus

    data_dir: string, the relative location of the data directory in this project
    filename: string, the name of the .RRF file
    col_names: list, the names of the columns in the .RRF file

    return: DataFrame, the data
    """
    # Put together the full filename
    load_file = os.path.join(data_dir, '2018AA-full/2018AA/META/', filename)

    # Read the file
    data = pd.read_csv(load_file, sep='|', header=None)

    # Lines end in pipe, so extra column will exist that needs to be dropped
    data = data.iloc[:, :-1]

    # Set the proper column names
    if col_names is None:
        col_names = get_colnames(filename, data_dir)
    data.columns = col_names

    return data


def get_colnames(filename, data_dir='../data'):
    # Column names for the file information file
    col_names = ['FIL', 'DES', 'FMT', 'CLS', 'RWS', 'BTS']
    metadata = read_rrf_file('MRFILES.RRF.gz', data_dir, col_names)

    # Filenames won't be gzipped in the metadata, so removed suffix
    if filename.endswith('.gz'):
        filename = filename[:-3]

    # Find the line for the given file and process the column names
    col_names_out = metadata.query('FIL == @filename')['FMT'].values[0]
    col_names_out = col_names_out.split(',')

    return col_names_out


def open_mrconso(data_dir='../data'):
    """
    Opens the Cocepts file from the UMLS Metahesaurus

    data_dir: string, relative location of the data directory in this project.

    return: DataFrame, the data contained int the file MRCONSO.RRF
    """
    return read_rrf_file('MRCONSO.RRF', data_dir)


def open_mrcui(data_dir='../data'):
    """
    Opens the CUI info file from the UMLS Metahesaurus

    data_dir: string, relative location of the data directory in this project.

    return: DataFrame, the data contained int the file MRCUI.RRF
    """
    return read_rrf_file('MRCUI.RRF.gz', data_dir)


def open_mrsty(data_dir='../data'):
    """
    Opens the semmantic type info file from the UMLS Metahesaurus

    data_dir: string, relative location of the data directory in this project.

    return: DataFrame, the data contained int the file MRSTY.RRF
    """
    return read_rrf_file('MRSTY.RRF.gz', data_dir)

