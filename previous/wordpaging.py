from win32com.client import Dispatch
import os
#open Word
dir = os.path.join('F:\scientificresearch',"documents","special talk","bowel sounds")
file = "jounral_presentation_lines.docx"
doc_path = os.path.join(dir,file)
word = Dispatch('Word.Application')
word.Visible = False
word = word.Documents.Open(doc_path)

#get number of sheets
word.Repaginate()
num_of_sheets = word.ComputeStatistics(2)
print(num_of_sheets)