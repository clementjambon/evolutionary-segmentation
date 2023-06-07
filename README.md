# Game Theory for Image Segmentation
*Clément JAMBON* 

## Experimenting with the main notebook

Before getting started, please make sure you have the necessary dependencies with
```bash
pip install -r requirements_dino.txt`
```

Then download the BSDS300 datasets (if not already done). To do so, we provide a script which you can execute with
```bash
cd scripts
./download_bsds.sh
```

You can then run the base notebook in `notebooks/gt_segmentation.ipynb`.

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