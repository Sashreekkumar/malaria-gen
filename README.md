# MalariaGen

## Progress
### Exploratory Data Analysis
- Checkout [MalariaGen_EDA.ipynb](notebooks/MalariaGen_EDA.ipynb)

### Project Files
- [data_loder.py](src/data_loader.py): Downloads genomic zip files one at a time, extracts only the 2L/calldata/GT/ folder for each sample, saves it as extracted/{sample_id}/gt/, then deletes the zip. Tracks progress by URL so interrupted runs resume correctly.

## Notes
My notes while studying for the required project
- [Anopheles.md](notes/Anopheles.md)
- [Genomics.md](notes/Genomics.md)
- [Malaria_Gen](notes/Malaria_Gen.md)