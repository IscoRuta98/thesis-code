## Detect.py
- A python script that uses pretrained Deep Learning models to detect the number of: figures, equations,and tables in a particular article.

### Assumptions
- Python is installed on your local machine.
- You have the following in your working directory:
  - `Detect.py` script
  - `requirements.txt` script
  - output.csv: Keeps records of the equations, figures, and tables for a particular article.
  - Master-list of a particular journal can be downloaded [here](https://drive.google.com/drive/folders/1Rnpm76i1vhD-lBplME15yQNZBxOv24Nk).
  - Pivot list of a particular journal can be downloaded [here](https://drive.google.com/drive/folders/1nmxCa9po1drhWBoMOxi1660XI3FzElEf).
  - Articles pertaining to a particular journal (You can download them [here](https://drive.google.com/drive/folders/1thYm5jauurF8m3Yd8AvZoZckO5G6mJDk)).
- You are either using Linux OS or macOS system. Dependencies such as [Deetectron2](https://github.com/facebookresearch/detectron2) (deep learning model used to detect tables and figures) are quite tricky to install on Windows operating systems 
  - [Click here](https://github.com/Layout-Parser/layout-parser/blob/main/installation.md) for additional Instruction to installing Detectron2 Layout Model for Windows users.

### Installation & Running `Detect.py` script

- In your working directory run the following command to install the python packages required for `Detect.py` script:
  - `pip install requirements.txt`
- In the `Detect.py` script, replace the following variables with the directory the files (i.e., master list, pivot, output Excel files, and directory to the articles on your local machine) are in:
```
masters = pd.read_excel("/working/directory/master.xlsx")
pivots = pd.read_excel("/working/directory/pivots.xlsx")
output = pd.read_csv("/working/directory/output.csv")
directory = r'/directory/to/stored/papers/
```

- To run the `Detect.py` script, run the following command in your terminal:
  - `python Detect.py`