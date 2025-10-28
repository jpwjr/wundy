# wundy
One dimension finite element program

## Install

### Clone repository

```console
git clone git@github.com:<user>/wundy
```

where `user` is your user name if you forked the repo, `tjfulle` otherwise.

### Create virtual environment

```console
python3 -m venv venv
source activate venv/bin/activate
```

### Install in editable mode

```console
cd wundy
python3 -m pip install -e .
```

## Test

In the `wundy` directory, execute

```console
pytest
```
## 📘 User Input Guide

The *wundy* solver uses structured YAML or JSON files to define models, materials, and loads.  
For a complete reference to input syntax and validation rules, see:

👉 [**User Input Specification →**](./docs/user_input_spec.md)

[![Documentation](https://img.shields.io/badge/docs-User_Input_Specification-blue?style=flat-square&logo=markdown)](./docs/user_input_specs.md)
