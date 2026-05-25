# Auto-extracted from colab_notebooks/0_prepare_grouped_dataset_for_training.ipynb cell 9.
# Keep notebook execute-only cells thin; edit behavior here.


def _read_gitignore_lines(path):
    for encoding in ("utf-8-sig", "utf-8", "cp1254", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.readlines(), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "unsupported .gitignore encoding")


# Fix .gitignore for prepared_runtime_datasets
gitignore_file = str(ROOT / ".gitignore")
lines, gitignore_encoding = _read_gitignore_lines(gitignore_file)

# Find the line with "data/prepared_runtime_datasets/*"
new_lines = []
i = 0
while i < len(lines):
    new_lines.append(lines[i])
    if lines[i].strip() == "data/prepared_runtime_datasets/*":
        # Add the next line (.gitkeep exception)
        if i + 1 < len(lines) and ".gitkeep" in lines[i + 1]:
            i += 1
            new_lines.append(lines[i])
        # Add our new exceptions if not already there
        if not any("!data/prepared_runtime_datasets/*/" in l for l in new_lines):
            new_lines.append("!data/prepared_runtime_datasets/*/\n")
            new_lines.append("!data/prepared_runtime_datasets/**/*\n")
    i += 1

# Write back
with open(gitignore_file, "w", encoding=gitignore_encoding) as f:
    f.writelines(new_lines)

print("OK: .gitignore fixed")
