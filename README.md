# PerceiverIO & Adversarial Attacks


## White box attacks

- Fast Gradient Sign Method (FGSM)
    ~~~
    python attacks/white_box/fgsm.py [--model {perceiver-io,vit}] [--epsilon EPSILON] [--size SIZE] [--batch BATCH]
    ~~~
    where size is the number of images we want to attack.

- Projected Gradient Descent (PGD)
Similarly,
    ~~~
    python attacks/white_box/pgd.py [-h] [--model {perceiver-io,vit}] [--epsilon EPSILON] [--step STEP] [--iterations ITERATIONS] [--size SIZE] [--batch BATCH]
    ~~~

## Black box attacks

### Transfer attacking
Use the adversarial examples that were found for a `source` model by using a white-box method in order 
to attack a `target` network. This is considered a black-box attack as we do not use the model parameters of the target network:

- FGSM:
    ~~~
    python attacks/black_box/fgsm_transfer_attack.py  [-h] [--source {perceiver-io,vit}] [-h] [--target {perceiver-io,vit}] [--epsilon EPSILON] [--size SIZE] [--batch BATCH]
    ~~~

- PGD:
    ~~~
    python attacks/black_box/pgd_transfer_attack.py  [-h] [--source {perceiver-io,vit}] [-h] [--target {perceiver-io,vit}] [--epsilon EPSILON] [--step STEP] [--iterations ITERATIONS] [--size SIZE] [--batch BATCH]
    ~~~


## Development Guidelines

### Setup environment (conda creation needs to be tested!)
In case conda environments are supported on the machine, run the following commands.
Otherwise, create a python environment and run the last two commands.
~~~
make conda-env
make activate-conda-env
make install
cp .env.local .env
~~~

Open `.env` file and add the environment variables you need.

### Download a new dataset
Go to `src/data/datasets.py` and add a new dataset name and its respective URL link:
- For Kaggle dataset, fill in the respected entry to the `KAGGLE_URLS`
- For a dataset stored in specific http link, fill in the respected entry to the `WGET_URLS`

Run the following script to download the dataset:
~~~
python -m src.data.scripts.download --dataset <dataset-name> --path <path-to-store-the-dataset>
~~~

where `<dataset-name>` is the name of the dataset provided in `src/data/datasets.py`.
The `path` argument takes a default value from `.env`, so there is no need to fill it in every time.

### Prepare a dataset for PyTorch DataLoader
Go to `src/data/scripts/prepare.py` and add a handler for the new dataset in order to create the structure requested for
PyTorch DataLoader.

For example if the `fruits/train` is given to the PyTorch DataLoader the structure of this directory
should be the below, where `orange` and `apple` are the classes of the dataset.
fruits/train
├── orange
│   ├── orange_image1.png
│   └── orange_image1.png
├── apple
│   └── apple_image1.png
│   └── apple_image2.png
│   └── apple_image3.png
~~~

After implementing the handler of the dataset run the below command:
~~~
python -m src.data.scripts.prepare --dataset <dataset-name> --path <path-to-store-the-dataset>
~~~

### Open locally jupyter lab from server
~~~
# for doris:
make jupyter-doris id=s3264009

# for boris:
make jupyter-boris id=s3264009
~~~

where `id` is the student id.

Then, open:
- `http://localhost:9001/lab` for `doris`
- `http://localhost:8999/lab` for `boris`


### Deploy on server
~~~
make deploy id=s3264009 machine=doris
~~~
where `id` is the student id and machine is either `doris` or `boris`

Note: The `.env` file should be configured on the server side.


### Download code from server
In case you want to download the current version of the code from server, use the following command:
~~~
make download id=s3264009 machine=doris
~~~
where `id` is the student id and machine is either `doris` or `boris`

