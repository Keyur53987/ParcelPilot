import pandas as pd

excel_path = "../AI Agent Assessment - Candidate Pack/ParcelPilot_Assessment_Data.xlsx"
xls = pd.ExcelFile(excel_path)
print("Sheets:", xls.sheet_names)

for sheet in xls.sheet_names:
    print(f"\n--- {sheet} ---")
    df = pd.read_excel(xls, sheet_name=sheet)
    print(df.head(2).to_markdown())
