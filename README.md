# TFG


VIRAT RELEASE 2.0 (12)
The file preprocessing_2.0.py takes as input the filepaths_2.0.txt (that contains all the path to all annotations and videos of VIRAT 2.0 Dataset). This function will process all txt files and generate trajectories_V0_2.0.json file. 

DIVA IARPA - Extended release (46)
The file preprocessing_Extended.py takes as input the filepaths_Extended.txt (that contains all the path to all annotations and videos of VIRAT 2.0 Dataset). This function will process all txt files and generate trajectories_V0_Extended.json file. 


## DATA VERSIONS
### TRAJECTORIES V0
This data has been extracted directly from DIVA IARPA (extended) and VIRAT 2.0 and put together in the same format. 
V0 – EXTENDED
Total of 7704 events
![image](https://github.com/user-attachments/assets/26094798-7769-421b-860c-3304af47eb19)
V0 – 2.0
Total of 1555 events
![image](https://github.com/user-attachments/assets/a45a7998-b5c3-4c81-944b-c1cfc4930f00)


### TRAJECTORIES V1
Filter out non relevant activities and keep only the ones that might be interesting

• Activities that were kept:  "Person loading an Object to a Vehicle",  "Person getting into a Vehicle", "Person Unloading an Object from a Car/Vehicle", "Person getting out of a Vehicle", "Person entering a facility", "Person exiting a facility", "Person running", "Exiting", "Opening", "Closing", "activity_standing", "vehicle_stopping", "activity_walking", "vehicle_turning_right", "vehicle_moving", "activity_carrying", "vehicle_starting", "vehicle_turning_left", "activity_running", "Unloading",  "Loading", "Entering", "vehicle_u_turn"


Merge data from both sources and create one unique Dataset V0. 

There are non-valid cases that lack data. Such invalid cases were filtered out based on this criterion:

•	Location information (actor’s bounding box) must be available for at least 10 frames (0.5 seconds).

After the filtering process we got the Dataset V1. 
Total of 4876 events 

![image](https://github.com/user-attachments/assets/3ea35441-b1d5-4267-b505-65eefb7ebc86)

### TRAJECTORIES V2
In V1 there are 974 samples for “vehicle_moving” activity. 
After analyzing the samples, I realized that it englobes many sub-activities within vehicle moving. 
![image](https://github.com/user-attachments/assets/24943990-0a36-40de-abae-376425527058)

Therefore, a manual relabeling process was done with these activities, visualizing each of the trajectories and assigning one of the 5 labels above (turning left, turning right, making u-turn, moving forward, moving backwards).
The number of vehicle moving samples went down significantly, from 974 to 427. 
With the relabeling process new samples were added to the existing vehicle turn activities. And two new activity-types were created: forward and backwards.

![image](https://github.com/user-attachments/assets/e8ac3bc2-432b-46a8-8a0c-b1006533b458)

### TRAJECTORIES V3
A subset of vehicle-related activities only was created, filtering out the rest of the activities: 
•	Activities that were kept: 
"Person getting into a Vehicle”, "Person getting out of a Vehicle", "Exiting", "vehicle_turning_right”, "vehicle_moving_forward", "vehicle_turning_left", "vehicle_moving_backward", "Entering", "vehicle_u_turn"

(Here entering and exiting refer to vehicles but also buildings, so it needs to be reviewed later in the manual cleaning process)

To have true negative samples for questions related to person entering or exiting the car, 150 “activity_walking” events were selected randomly, where the person is simply walking. 

![image](https://github.com/user-attachments/assets/0f85cb1c-3af8-4280-99b5-056588e17194)

### TRAJECTORIES V4
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
Once the final set is created, with the correct questions per image, I went through a manual cleaning process, and in case a question was not adequate for a case, it was removed. 
The number of questions after the cleaning process decreased from 2220 to 1874 questions, so 346 prompts were removed.


The removed prompts are in this file: https://milestonesys365-my.sharepoint.com/personal/norm_milestone_dk/Documents/Desktop/NORA%20THESIS%20PROJECT/ALL%20DATA/DATASET%20VERSIONS/excluded_cases_combined.tsv



The final, cleaned version of the dataset with all the information to generate both  visual and textual prompt can be found in this file: https://milestonesys365-my.sharepoint.com/personal/norm_milestone_dk/Documents/Desktop/NORA%20THESIS%20PROJECT/ALL%20DATA/DATASET%20VERSIONS/trajectories_V4.tsv




