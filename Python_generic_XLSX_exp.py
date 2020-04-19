##WILL BE SAVED IN THE SAME FOLDER AS THIS SCRIPT
# Create a Pandas Excel writer using XlsxWriter as the engine.
sheet_name = 'Sheet_1'
writer     = pd.ExcelWriter('filename.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name=sheet_name)

# Access the XlsxWriter workbook and worksheet objects from the dataframe.
workbook  = writer.book
worksheet = writer.sheets[sheet_name]

# Adjust the width of the first column to make the date values clearer.
worksheet.set_column('A:A', 20)

# Close the Pandas Excel writer and output the Excel file.
writer.save()