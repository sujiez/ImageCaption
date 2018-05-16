# ImageCaption
Need to download MSCOCO caption dataset and use process_coco_data.py and generate_val_small.py to preprocess data to TFRecord.

Need to clone [this](https://github.com/tylin/coco-caption) in the same directory for evaluation. 

All configuration is in configure.py

Run train_solver.py to train the model

Run test_wrapper.py to run test and give Bleu score.

