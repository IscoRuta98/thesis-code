# Import libraries
import cv2
import layoutparser as lp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileReader, PdfFileWriter
from pdf2image import convert_from_path
import os
import time

# Read in masters and pivot list
masters = pd.read_excel("/home/isco/Documents/Master lists/JPE_master.xlsx")
pivots = pd.read_excel("/home/isco/Documents/Pivot lists/JPE_pivots.xlsx")
output = pd.read_csv("/home/isco/Documents/thesis-code/datadump.csv")


# Create Dataframe
"""cols2 = ['stable_url', 'authors', 'title', 'abstract', 'content_type', 'issue_url', 'pages', 'no_tables', 'no_eq',
         'no_figures']
output = pd.DataFrame(columns=cols2)
"""
# directory path of the papers
directory = r'/home/isco/Documents/SamplePapers/'

# Load the deep layout models from the layoutparser API
model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"})

formula_model = lp.Detectron2LayoutModel(
    'lp://MFD/faster_rcnn_R_50_FPN_3x/config',
    label_map={1: "Equation"})

total = 0
fulllist_s_1940 = 0
for ind in pivots.index:
    temp3 = masters[masters['issue_url'] == pivots['issue_url'][ind]]
    downloaded = 0
    for ind2 in temp3.index:
        pdf_file_name = masters['stable_url'][ind2].split('/')[-1] + ".pdf"
        if (pivots['year'][ind] >= 1940 and pivots['year'][ind] <= 2010):
            fulllist_s_1940 += 1
            # print(pdf_file_name+" "+str(os.path.isfile(directory+pdf_file_name)))
            check = output[['stable_url']].isin({'stable_url': [masters['stable_url'][ind2]]}).all(1).any()
            if (check):
                continue
            check2 = os.path.isfile(directory + "/" + pdf_file_name)

            if (check2):
                downloaded += 1

                pdf_document = directory + "/" + pdf_file_name
                print("Checking Article: ", pdf_file_name)
                pdf = PdfFileReader(pdf_document)
                images = convert_from_path(pdf_document)

                # Set up a counter to keep track.
                NumOfTables = 0
                NumOfFigures = 0
                NumOfEquation = 0
                start = time.time()

                for i in range(pdf.getNumPages()):
                    if(i == 0):
                        continue
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
                    Table = df.loc[df['type'] == 'Table']
                    Figure = df.loc[df['type'] == 'Figure']
                    # Equation = df2[df2['type'] == 'Equation']

                    NumOfTables = NumOfTables + len(Table)
                    NumOfFigures = NumOfFigures + len(Figure)
                    NumOfEquation = NumOfEquation + len(df2)

                print("The numer of Tables in this article is: ", NumOfTables)
                print("The numer of Figures in this article is: ", NumOfFigures)
                print("The numer of Equations in this article is: ", NumOfEquation)

                temp = {'stable_url': masters['stable_url'][ind2], 'authors': masters['authors'][ind2], 'title': masters['title'][ind2],
                        'abstract': masters['abstract'][ind2], 'content_type': masters['content_type'][ind2],
                        'issue_url': masters['issue_url'][ind2], 'pages': masters['pages'][ind2],
                        'no_tables': NumOfTables, 'no_eq': NumOfEquation, 'no_figures': NumOfFigures}

                output = output.append(temp, ignore_index=True)
                output.to_csv('/home/isco/Documents/thesis-code/datadump.csv', index=False)

                end = time.time()
                print(str(pdf_file_name) + " took " + str(np.round((end - start) / 60,2)) + " minutes to process this article")
                print("Next article...")
            # do parsing processing for this paper
    total = total + downloaded
    """if (pivots['year'][ind] >= 1940 and pivots['year'][ind] <= 2010):
        print(
        str(pivots['year'][ind]) + " " + str(pivots['no_docs'][ind]) + " " + str(pivots['issue_url'][ind]) + " " + str(
            downloaded))"""
print(str(total) + " article processed in this session")
print(fulllist_s_1940)
