# Code Part (Paul Ingram Project)

This repository contains the clean implementation scripts extracted from the project folder `Paul Ingram_25.2.21`.

Only the `Code Part` subdirectory is tracked in this public repository. All other folders (e.g., raw data, annotations, analysis phases) are excluded for clarity and data privacy.

## Folder Structure
Code Part/
├── ImageProcessor2.py
├── PPTXProcessor.py
└── ...


## Dependencies

Please note that `ImageProcessor2.py` is a dependency of `PPTXProcessor.py`.  
To run the script, you will need to specify:

- The path to the `respondent_igo` file  
- The path to the document being processed (e.g., `.pptx`, `.docx`)

### Example usage

```python
from PPTXProcessor import process_pptx
process_pptx(file_path="path/to/pptx", igo_path="path/to/respondent_igo.xlsx")
```

