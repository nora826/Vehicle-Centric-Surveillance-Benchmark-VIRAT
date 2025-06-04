All the information on the dataset generation is explained in _DataVersions.md_ file. 

## Customise dataset
In order to dowload the dataset as desired: 

streamlit run configurations_dataset.py

By running this command it will open a webapp where the desired configuration can be chosen: 
- which questions to include
- including negative (none of the above) cases
- what image layout to choose (overlaying trajectory, bbox, blur, crop...)
- what information to include in the data.tsv file appart from the main information

After choosing the desired configuration, by pressing "Generate Dataset With This Configuration", it will create a new directory where the customised dataset is stored.

## Testing Models
