# MLBaryonPainting
Learning to paint cosmo sim outputs onto N-body maps

Paper: https://arxiv.org/abs/2307.16733. Accepted to MNRAS Aug 24, 2023.

Trained models: https://www.dropbox.com/sh/29v4js55hc1gais/AAC1U-dW1PRjvjF7Mjofsli5a?dl=0 

----
# Generating training images

- First, create a catalog of FoF halos that meet your criteria. Use [make_cluster_catalog.py](make_cluster_catalogs.py).
- Next, create cutouts of these halos. If you have snapshots of TNG stored locally, use [cutouts_from_fullbox.py](cutouts_from_fullbox.py). Else, use [cutouts_via_api.py]((cutouts_via_api.py). In the latter case, don't forget to first register for an account at https://www.tng-project.org/data/ and modify the "api_key" field in this file with your API key. The default value of 12345 will *not* work.
- Import [make_yt_proj.py](make_yt_proj.py). This contains the `yt_xray` function that will read in your HDF5 snapshots and output FITS images.
- Run [compile_training_data.py](compile_training_data.py). First, this compiles the FITS images into a single array for each property. Then, it normalises each array two ways. It stores the normalisation parameters so you can revert the operation later.

# Train a model!
- Set up your computing environment to activate GPUs and load `keras` and `tensorflow`. This varies widely from computing cluster to cluster, but an example using SLURM on the Harvard compute cluster is shown in start_tensorflow.sh.
- Next, run [autoencoder.py](autoencoder.py)! You can vary the input and output files as desired.

# Assess performance of trained model
You can find a variety of tools to assess your trained model in [compare_trained_models.py](compare_trained_models.py). Essentially, you load the trained model and apply it to the test data (which the model has never seen). Then you can visualize the predictions and compare to the ground truth in a variety of ways. 
