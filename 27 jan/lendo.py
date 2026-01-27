from pathlib import Path
import pandas as pd

xlsx_path = Path(r"C:\Users\Hézio\Pictures\IC\pendulo pasco-longo.xlsx")

xls = pd.ExcelFile(xlsx_path)
print("Abas encontradas:", xls.sheet_names)

out_dir = xlsx_path.parent
for sheet in xls.sheet_names:
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    safe_sheet = "".join(c if c.isalnum() or c in " _-" else "_" for c in sheet).strip()
    out_csv = out_dir / f"{xlsx_path.stem}__{safe_sheet}.csv"
    df.to_csv(out_csv, index=False, sep=";", encoding="utf-8-sig", decimal=",")
    print("Gerado:", out_csv)

print("Fim.")
