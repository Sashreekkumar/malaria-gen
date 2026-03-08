import nbformat

notebook_path = "your_notebook.ipynb"

# Load the notebook
nb = nbformat.read(notebook_path, as_version=5)

# Fix widgets metadata
for cell in nb.cells:
    if "metadata" in cell and "widgets" in cell["metadata"]:
        for key, widget in cell["metadata"]["widgets"].items():
            if "state" not in widget:
                widget["state"] = {}  # Add empty state

# Save back
nbformat.write(nb, notebook_path)
print("Notebook fixed.")