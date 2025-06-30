## initial Processing
All the information on the dataset generation is explained in _DataVersions.md_ file. 

## Benchmark Dataset
In order to dowload the dataset as desired: 

    streamlit run configurations_dataset.py

By running this command it will open a webapp where the desired configuration can be chosen: 
- which questions to include
- including negative (none of the above) cases
- image layout (overlaying trajectory, bbox, blur, crop...)
- what information to include in the data.tsv file appart from the main information

After choosing the desired configuration, by pressing "Generate Dataset With This Configuration", it will create a new directory where the customised dataset is stored.


The configurations_dataset.py file reads raw-dataset directory, that contains images and data.tsv

## Testing

To test the model with the desired configurations: 

    python internVL.py path/to/config/directory
    python internVL.py path/to/config/directory

To get the evaluation metrics (accuracy, F1-Score) and the cofussion matrices per question: 
    python evaluation.py path/to/config/directory/internVL_results.tsv
    python evaluation.py path/to/config/directory/qwen_2B_results.tsv
