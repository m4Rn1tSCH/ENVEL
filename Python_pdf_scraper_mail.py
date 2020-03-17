# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:39:28 2020

@author: bill-
"""

from io import StringIO
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
import re

def get_cv_email(self, cv_path):
    pagenums = set()
    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)
    infile = open(cv_path, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close()
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    email = match.group(0)
    return email

#\w+(?:[.-]\w+)*@\w+(?:[.-]\w+)+[.-][a-z_0-9]+(?=[A-Z]|(?!=[.-])\b)