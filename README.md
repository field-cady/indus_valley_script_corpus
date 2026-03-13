# Indus Valley Script Corpus

This directory contains a machine-readable version of the Indus Valley Script corpus, primarily derived from the **Interactive Corpus of Indus Texts (ICIT)** and the foundational work of **Iravatham Mahadevan**.

## Data Source
The data was retrieved from the **Lipi Repository** (maintained by yajnadevam), which provides open-source CSV exports of the ICIT and Mahadevan corpora.

- **Primary Repository:** [yajnadevam/lipi](https://github.com/yajnadevam/lipi)
- **Academic Context:** The ICIT is a "living corpus" managed by Dr. Andreas Fuls and Bryan K. Wells, providing the most up-to-date catalog of Indus inscriptions.

## File Descriptions

### `inscriptions.csv`
The primary dataset containing over 5,000 inscriptions.
- **`id` / `cisi`**: Standard identifiers for the artifact (Corpus of Indus Seals and Inscriptions).
- **`site` / `region`**: Geographical origin (e.g., Harappa, Mohenjo-Daro).
- **`text`**: The inscription itself, represented as a sequence of sign numbers (e.g., `+410-017+`).
- **Metadata**: Material, shape, preservation status, and excavation details.

### `words.csv`
A collection of identified sign sequences or "word" candidates derived from statistical analysis of the corpus.

### `xlits.csv`
Transliteration mappings and sign numbering cross-references. This is essential for converting the numeric sign codes into different numbering systems (like Mahadevan’s 417-sign list).

## Sign Numbering
The numeric codes in the `text` column refer to the standardized sign list used by the ICIT. For example, sign `740` typically refers to the "jar" sign, one of the most common symbols in the script.

## Usage
These files are formatted for direct use with data analysis tools like Python (Pandas) or R.
