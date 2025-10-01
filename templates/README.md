# Data Templates

The Streamlit app can generate ready-to-use workbooks that match the input format expected by the loaders.
Use the download button in **Upload → Households** to grab a 2022, 15-minute template with the full timestamp
column already populated (Europe/Rome timezone aware).

If you prefer to build the workbook from the command line, you can run the helper exposed in `src.exporters`:

```python
from pathlib import Path
from src.exporters import build_household_template

Path("households_template_2022.xlsx").write_bytes(build_household_template())
```

Each worksheet represents one household (e.g., `HH01`). Keep exactly two columns:

- `timestamp`: 15-minute local timestamps (Europe/Rome). Leave them as Excel datetimes; the loader will take care of
  daylight-saving transitions when aggregating to hours.
- `Power_kW`: measured power demand in kW over each 15-minute interval.

Duplicate the worksheet for every household you need to model and replace the placeholder values with your own
measurements. Do not add worksheets without the required columns; the importer reads every sheet in the workbook.
