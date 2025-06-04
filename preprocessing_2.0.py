import os
import csv
import json
from collections import defaultdict

# Mapping dictionaries for event and object types
EVENT_TYPE_MAP = {
    "1": "Person loading an Object to a Vehicle",
    "2": "Person Unloading an Object from a Car/Vehicle",
    "3": "Person Opening a Vehicle/Car Trunk",
    "4": "Person Closing a Vehicle/Car Trunk",
    "5": "Person getting into a Vehicle",
    "6": "Person getting out of a Vehicle",
    "7": "Person gesturing",
    "8": "Person digging",
    "9": "Person carrying an object",
    "10": "Person running",
    "11": "Person entering a facility",
    "12": "Person exiting a facility"
}

OBJECT_TYPE_MAP = {
    "1": "person",
    "2": "car",
    "3": "vehicles",
    "4": "object",
    "5": "bike/bicycles"
}

def load_event_file(event_file_path):
    """Reads the event file and returns a list of rows (as lists of strings)."""
    events = []
    with open(event_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            events.append(parts)
    return events

def load_object_file(object_file_path):
    """
    Reads the object file and returns a dictionary mapping object_id to a list of rows.
    Each row is stored as a dict.
    """
    objects = defaultdict(list)
    with open(object_file_path, "r") as f:
        reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
        for row in reader:
            if not row or len(row) < 8:
                continue
            obj = {
                'object_id': row[0],
                'duration': int(row[1]),
                'frame': int(row[2]),
                'bbox_lefttop_x': float(row[3]),
                'bbox_lefttop_y': float(row[4]),
                'bbox_width': float(row[5]),
                'bbox_height': float(row[6]),
                'object_type': row[7]
            }
            objects[row[0]].append(obj)
    return objects

def load_mapping_file(mapping_file_path):
    """
    Reads the mapping file and returns a list of rows (each a list of strings).
    """
    mappings = []
    with open(mapping_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            mappings.append(parts)
    return mappings

def build_ordered_object_ids(object_file_path):
    """
    Returns an ordered list of unique object IDs as they first appear in the object file.
    """
    ordered_ids = []
    with open(object_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            obj_id = parts[0]
            if obj_id not in ordered_ids:
                ordered_ids.append(obj_id)
    return ordered_ids

def extract_bbox_data_for_object(object_rows, start_frame, end_frame):
    """
    Extracts bounding box info from the object rows for frames between start_frame and end_frame.
    Computes mid_x and mid_y.
    Returns a list of bbox dictionaries sorted by frame.
    """
    bbox_list = []
    for row in object_rows:
        frame = row['frame']
        if start_frame <= frame <= end_frame:
            x = row['bbox_lefttop_x']
            y = row['bbox_lefttop_y']
            w = row['bbox_width']
            h = row['bbox_height']
            bbox_list.append({
                'frame': frame,
                'bbox_lefttop_x': x,
                'bbox_lefttop_y': y,
                'bbox_width': w,
                'bbox_height': h,
                'mid_x': x + w/2,
                'mid_y': y + h/2
            })
    return sorted(bbox_list, key=lambda d: d['frame'])

def generate_event_dicts(event_file_path, mapping_file_path, object_file_path, video_source):
    """
    Processes the three files and returns a dictionary of event dictionaries.
    The key is constructed as <base_id>_<activity_id>, where base_id is the portion
    after "VIRAT_S_" in the video_source.
    """
    mapping_rows = load_mapping_file(mapping_file_path)
    objects_dict = load_object_file(object_file_path)
    ordered_obj_ids = build_ordered_object_ids(object_file_path)
    
    events_dict = {}
    # Extract base id from video_source: e.g. from ".../VIRAT_S_000001" get "000001"
    base_id = ""
    if "VIRAT_S_" in video_source:
        base_id = video_source.split("VIRAT_S_")[-1]
    
    for mapping in mapping_rows:
        # Mapping file columns:
        # 0: event_id, 1: event_type, 2: duration, 3: start_frame, 4: end_frame,
        # 5: number_of_objects, 6 onward: association flags ("1" or "0")
        event_id = mapping[0]
        event_type = mapping[1]
        start_frame = int(mapping[3])
        end_frame = int(mapping[4])
        activity_label = EVENT_TYPE_MAP.get(str(event_type), "Unknown")
        
        trajectory = []
        # Iterate over association flags (starting at column 6)
        for idx, flag in enumerate(mapping[6:]):
            if flag == "1":
                if idx < len(ordered_obj_ids):
                    obj_id = ordered_obj_ids[idx]
                    obj_rows = objects_dict.get(obj_id, [])
                    bbox_per_frame = extract_bbox_data_for_object(obj_rows, start_frame, end_frame)
                    # Use the first occurrence to determine actor type, if available
                    if obj_rows:
                        obj_type = OBJECT_TYPE_MAP.get(obj_rows[0]['object_type'], "Unknown")
                    else:
                        obj_type = "Unknown"
                    trajectory.append({
                        'actor_id': obj_id,
                        'actor_type': obj_type,
                        'bbox_per_frame': bbox_per_frame
                    })
        
        event_dict = {
            'event_id': event_id,
            'activity': activity_label,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'trajectory': trajectory,
            'activity_id': event_id,
            'source': os.path.abspath(video_source)
        }
        # Create dictionary key as "<base_id>_<activity_id>"
        key = f"{base_id}_{event_dict['activity_id']}"
        events_dict[key] = event_dict
    
    return events_dict

def process_annotation_list(list_file_path):
    """
    Reads a file containing a list of base annotation paths (one per line).
    For each path, it appends the expected filename suffixes for events, mapping, and objects,
    processes them, and combines the resulting event dictionaries into one dictionary.
    The keys of the dictionary are constructed as <base_id>_<activity_id>.
    """
    combined_events = {}
    with open(list_file_path, "r") as f:
        for line in f:
            base_path = line.strip()
            if not base_path:
                continue
            
            # Construct full file paths by appending suffixes
            event_file = base_path + ".viratdata.events.txt"
            mapping_file = base_path + ".viratdata.mapping.txt"
            object_file = base_path + ".viratdata.objects.txt"
            
            # Check if all files exist; if not, skip this entry.
            if not (os.path.exists(event_file) and os.path.exists(mapping_file) and os.path.exists(object_file)):
                print(f"Skipping {base_path} because one or more annotation files are missing.")
                continue
            
            # Here, we assume video_source is the same as the base_path.
            video_source = base_path
            
            # Generate events for the current annotation set
            events = generate_event_dicts(event_file, mapping_file, object_file, video_source)
            # Merge into the combined dictionary
            combined_events.update(events)
    return combined_events


if __name__ == "__main__":
    # Path to the file that lists all base annotation paths
    list_file_path = "/home/norm/workspace/CameraAsWitness/dataset2/filepaths_2.0.txt"
    
    # Process the list and generate a combined list of event dictionaries
    combined_events = process_annotation_list(list_file_path)
    
    # Write the combined events to a JSON file or print them
    output_json = json.dumps(combined_events, indent=2)
    #(output_json)
    
    # Optionally, save to a file:
    with open("trajectories_2.0.json", "w") as outfile:
        outfile.write(output_json)
