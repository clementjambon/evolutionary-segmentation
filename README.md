# Game Theory for Image Segmentation
*Clément JAMBON* 

This is the implementation of a small project I did for the block course ["Controversies in Game Theory IX: Cooperative and Non-​Cooperative Game Theory"](https://coss.ethz.ch/education/controversies.html).

## Running the notebooks

Before getting started, please make sure you have the necessary dependencies with
```bash
pip install -r requirements.txt`
```

Then download the BSDS300 datasets (if not already done). To do so, we provide a script which you can execute with
```bash
cd scripts
./download_bsds.sh
```

You can then run the base notebook in `notebooks/gt_segmentation.ipynb`.

If you want to reproduce the results with the DINO features, you will need to extract them as described in the next paragraph and then use the dedicated notebook in `notebooks/dino_segmentation.ipynb`.

## Extracting DINO features

In order to play with DINO features, first make sure you installed the necessary modules with
```bash
pip install -r requirements_dino.txt`
```

Then, run (this might take some time)
```bash
cd scripts
python extract-dino.py
```

Note that by default, features are extracted in `N_TILESxN_TILES` tiles with `N_TILES=2`. The value can be changed within `scripts/extract_dino.py`.