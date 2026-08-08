# Data

Raw and processed data are intentionally excluded from Git.

Official dataset: Bormann et al., **FORCE 2020 Well Log and Lithofacies
Dataset for Machine Learning Competition**, Zenodo.
DOI: https://doi.org/10.5281/zenodo.4351156

The original well logs are supplied under the Norwegian Licence for Open
Government Data (NLOD) 2.0. Any publication using the lithofacies data must
include the attribution required by the FORCE competition:

> Lithofacies data was provided by the FORCE Machine Learning competition
> with well logs and seismic 2020.

Download the LAS archive from Zenodo, extract it under `data/raw/force2020`,
then run:

```bash
lithology-unet prepare-las data/raw/force2020 \
  --output data/processed/force2020.csv
```

The converter replaces the LAS null value (`-999.25`) with missing values and
retains only labelled rows. Do not commit generated tables.
