# PyHillFit data format

This folder contains data from the paper:
[Crumb et al. Journal of Pharmacological & Toxicological Methods (2016): An evaluation of 30 clinical drugs against the CiPA proposed ion channel panel](https://www.ncbi.nlm.nih.gov/pubmed/27060526)

The tables have comma separated values with this format on each line (plus a header line):
['Drug','Channel','Experiment','Concentration(uM)','%Inhibition']

## Sub-folder
The sub-folder contains some scripts that were used to generate these .csv files from the raw data in the original publication.

Raw data is in in `Spreadsheet_of_data.xlsx` (Excel format) which is then exported to .csv

It is analysed by running 'assemble_python_input_file.m' to create the file
```
crumb_data.csv
```

It can be loaded, for instance, by using 'pandas' in the file [Crumb_data/Crumb_data_loading.py](Crumb_data/Crumb_data_loading.py).
