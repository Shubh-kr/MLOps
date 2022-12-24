### Introduction

The idea of this repo is to understand the basics of MLOps for personal understanding (Referenced from: https://www.ravirajag.dev/blog/). This will cover the basics of MLOps:
- Model Building
- Monitoring
- Configurations
- Testing
- Packaging
- Deployment
- CI/CD

A basic classification task (NLP based).

### Dataset
CoLA (Corpus of Linguistic Acceptability) database.

### Data Pipelines

Data Pipelines are created using Pytorch Lighning --> DataModulues (alternative, Vanilla Pytorch --> DatLoaders).

_DataModules_ are structured definition, which allows for additional optimizations such as automated distribution of workload between CPU and GPU. Using _Datamodules_ is recommended whenever possible.

A DataModule is definedby an interface;
- **prepare_data** (optional) which is called only once an on 1 GPU -- typically something like the data download step we have below
- **setup**, which is called on each GPu separately and accepts **stage** to define if we are a **fit** or **test** step.
- **train_dataloader**, **val_dataloader** and **test_dataloader** to load each dataset.

A DataModule encapsulated the five steps involded in data processing in PyTorch:
- Download/ tokenize/ process
- Clean and (maybe) save to disk
- Load inside Dataset.
- Apply transforms (rotate, tokenize, etc..).
- Wrap inside a DataLoader.
