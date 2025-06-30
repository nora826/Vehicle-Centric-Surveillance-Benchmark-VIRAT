## Initial Processing



##### VIRAT RELEASE 2.0 (12)
The file _preprocessing_2.0.py_ takes as input the _filepaths_2.0.txt_ (that contains all the path to all annotations and videos of VIRAT 2.0 Dataset). This function will process all txt files and generate _trajectories_V0_2.0.json_ file. 

    python preprocessing_2.0.py filepaths_2.0.txt

##### DIVA IARPA - Extended release (46)
The file _preprocessing_Extended.py_ takes as input the _filepaths_Extended.txt_ (that contains all the path to all annotations and videos of VIRAT 2.0 Dataset). This function will process all txt files and generate _trajectories_V0_Extended.json_ file. 

     python preprocessing_Extended.py filepaths_Extended.txt


### TRAJECTORIES V0
Merge data from both sources and create one unique Dataset V0. 

##### V0 – EXTENDED
![image](https://github.com/user-attachments/assets/26094798-7769-421b-860c-3304af47eb19)

##### V0 – 2.0
![image](https://github.com/user-attachments/assets/a45a7998-b5c3-4c81-944b-c1cfc4930f00)


### TRAJECTORIES V1
There are non-valid cases that lack data. Such invalid cases were filtered out based on this criterion:

•	Location information (actor’s bounding box) must be available for at least 10 frames (0.5 seconds).

After the filtering process we got the Dataset V1. 

### TRAJECTORIES V2
Filter out non relevant activities and keep only the ones that might be interesting

• Activities that were kept:  "Person loading an Object to a Vehicle",  "Person getting into a Vehicle", "Person Unloading an Object from a Car/Vehicle", "Person getting out of a Vehicle", "Person entering a facility", "Person exiting a facility", "Person running", "Exiting", "Opening", "Closing", "activity_standing", "vehicle_stopping", "activity_walking", "vehicle_turning_right", "vehicle_moving", "activity_carrying", "vehicle_starting", "vehicle_turning_left", "activity_running", "Unloading",  "Loading", "Entering", "vehicle_u_turn"


### TRAJECTORIES V3
In V1 there are 974 samples for “vehicle_moving” activity. 
Since it englobes many sub-activities within vehicle moving, relabelling process is done. 
![image](https://github.com/user-attachments/assets/24943990-0a36-40de-abae-376425527058)

A manual relabeling process was done with these activities, visualizing each of the trajectories and assigning one of the 5 labels above (turning left, turning right, making u-turn, moving forward, moving backwards). Python file using streamlit for relabelling is called _manual_labelling.py_.

    streamlit run manual_labelling.py 
    
With the relabeling process new samples were added to the existing vehicle turn activities. And two new activity-types were created: forward and backwards.


### TRAJECTORIES V4
A subset of vehicle-related activities only was created, filtering out the rest of the activities: 
•	Activities that were kept: 
"Person getting into a Vehicle”, "Person getting out of a Vehicle", "Exiting", "vehicle_turning_right”, "vehicle_moving_forward", "vehicle_turning_left", "vehicle_moving_backward", "Entering", "vehicle_u_turn"

To have true negative samples for questions related to person entering or exiting the car, 150 “activity_walking” events were selected randomly, where the person is simply walking. 


### TRAJECTORIES V4
Small subset with 8 activties only, for question generation.



Question-Image paired dataset is created using _generate_raw_dataset.py_ and introducing last dataset version JSON file as input.

    python generate_raw_dataset.py trajectories_V4.json
    
When generating the prompts these criteria were followed: 
Each image is paired with at least one question, either a three-option multiple choice or a binary yes/no question. 

The four question templates are:
1.	What turn is the vehicle making?
A) Left B) Right C) None of the above
2.	In which direction is the vehicle moving?
A) Forward B) Backward C) None of the above
3.	What is the person doing?
A) Entering the vehicle B) Exiting the vehicle C) None of the above
4.	Is the vehicle making a U-turn?
A)	Yes B) No 

Negative (“None of the above”) examples
When deciding how to include True negative samples, I considered that a vehicle could be performing multiple activities at the same time.

Taking such information into account, this is the method followed to generate real ”None of the above” samples:
•	Turn questions. Clips that show a vehicle moving straight forward are used, so the correct answers are C) None of the above for question 1 and B) No for question 4.
•	Direction question. Clips of a person entering or exiting a stationary vehicle are paired with question 2 because the car is not moving.
•	Person-action question. 150 randomly selected person walking trajectories are paired with Q3, where the correct answer is C) None of the above.

This strategy guarantees that every question type includes true negative cases.

Once the final set is created, with the correct questions per image, I went through a manual cleaning process  _data_cleaning.py_ , and in case a question was not adequate for a case, it was removed. 

    streamlit run cleaning.py


The final, cleaned version of the dataset with all the information to generate both visual and textual prompt can be found in TFG/Benchmark Dataset/Raw-Data




## Benchmark Dataset
In order to download the dataset as desired: 

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

The evaluation results are stored in config_dataset directory: 

    config_datasets/plots/ #confusion matrices
    config_datasets/evaluation_metrics.csv
    
    
