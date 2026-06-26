import pypandoc
pypandoc.download_pandoc()

# Define your input and output file paths
input_file = "Final_Report_Rewritten.md"
output_file = "document.docx"

# Convert the file directly
pypandoc.convert_file(input_file, 'docx', outputfile=output_file)

print(f"Successfully converted {input_file} to {output_file}!")
