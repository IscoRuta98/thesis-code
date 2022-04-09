# Import libraries
import cv2
import layoutparser as lp
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileReader, PdfFileWriter
from pdf2image import convert_from_path

# Load the deep layout models from the layoutparser API
model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"})

formula_model = lp.Detectron2LayoutModel(
    'lp://MFD/faster_rcnn_R_50_FPN_3x/config',
    label_map={1: "Equation"})

# Read in the PDF

pdf_document = "559.pdf"
pdf = PdfFileReader(pdf_document)
images = convert_from_path(pdf_document)

# Set up a counter to keep track.
NumOfTables = 0
NumOfFigures = 0
NumOfEquation = 0

for i in range(pdf.getNumPages()):
    """
    For every page in an article, two Deep Learning models are ran.
    One DL model detects: figures, and tables.
    The other model detects equations in a page.
    
    Once the above is detected, there is a counter that keeps track of:
    - The number of tables.
    - THe number of equations.
    - The number of figures.
    """
    layout = model.detect(images[i])
    EquationLayout = formula_model.detect(images[i])
    # Export layout results as a dataframe.
    df = layout.to_dataframe()
    df2 = EquationLayout.to_dataframe()
    # Extract the number of layouts (i.e. rows in the dataframe) that detected a table.
    Table = df[df['type'].str.contains('Table')]
    Figure = df[df['type'].str.contains('Figure')]
    # Equation = df2[df2['type'] == 'Equation']

    NumOfTables = NumOfTables + len(Table)
    NumOfFigures = NumOfFigures + len(Figure)
    NumOfEquation = NumOfEquation + len(df2)

print("The numer of Tables in this image is: ", len(NumOfTables))
print("The numer of Figures in this image is: ", len(NumOfFigures))
print("The numer of Equations in this image is: ", len(NumOfEquation))
