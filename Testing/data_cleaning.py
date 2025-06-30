import streamlit as st
import cv2
import numpy as np
import os
import json
import re
from image_generation import crop_video, visualize_trajectory_img, draw_bbox
from prompt_generation import generate_prompt

TRAJECTORIES_FILE = "/home/norm/workspace/CameraAsWitness/dataset/trajectories_V3.json" 





def load_trajectories_data(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def extract_trajectory(event_data):
    trajectories = {}
    for actor in event_data.get('trajectory', []):
        actor_id = actor['actor_id']
        trajectories[actor_id] = {
            'actor_type': actor['actor_type'],
            'frames': [],
            'x': [],
            'y': []
        }
        for bbox in actor.get('bbox_per_frame', []):
            trajectories[actor_id]['frames'].append(bbox['frame'])
            if 'mid_x' in bbox and 'mid_y' in bbox:
                trajectories[actor_id]['x'].append(bbox['mid_x'])
                trajectories[actor_id]['y'].append(bbox['mid_y'])
            elif all(k in bbox for k in ['bbox_lefttop_x', 'bbox_lefttop_y', 'bbox_width', 'bbox_height']):
                x = float(bbox['bbox_lefttop_x']) + float(bbox['bbox_width']) / 2
                y = float(bbox['bbox_lefttop_y']) + float(bbox['bbox_height']) / 2
                trajectories[actor_id]['x'].append(x)
                trajectories[actor_id]['y'].append(y)
            else:
                st.warning(f"Bounding box missing required keys: {bbox}")
    return trajectories



def go_to_next_case():
    event_keys = list(trajectories_data.keys())
    if st.session_state.selected_case_index < len(event_keys) - 1:
        st.session_state.selected_case_index += 1
    else:
        st.session_state.selected_case_index = 0


def remove_case():
    trajectories_data.pop(selected_event_key)

    with open(TRAJECTORIES_FILE, "w") as f:
        json.dump(trajectories_data, f, indent=2)
    
    event_keys = list(trajectories_data.keys())
    if not event_keys:
        st.error("No events remaining.")
        st.stop()
    if st.session_state.selected_case_index >= len(event_keys):
        st.session_state.selected_case_index = 0

    go_to_next_case()




trajectories_data = load_trajectories_data(TRAJECTORIES_FILE)
event_keys = list(trajectories_data.keys())

if not event_keys:
    st.error("No events found in the trajectories file.")
    st.stop()

if 'selected_case_index' not in st.session_state:
    st.session_state.selected_case_index = 0

case_number = st.sidebar.slider("Select Case", 1, len(event_keys), value=st.session_state.selected_case_index + 1, step=1)
st.session_state.selected_case_index = case_number - 1

selected_event_key = event_keys[st.session_state.selected_case_index]
selected_event = trajectories_data[selected_event_key]




# Main App Layout

st.title("Trajectory Visualization App")
st.write("**Selected Event Key:**", selected_event_key)

if "start_frame" not in selected_event or "end_frame" not in selected_event:
    st.error("Selected event is missing 'start_frame' or 'end_frame' information.")
    st.stop()

trajectories = extract_trajectory(selected_event)

video_path = selected_event.get("source", "")
video_path = selected_event.get("source", "").replace("/home/norm/data/", "/home/norm/workspace/")
if "/home/norm/workspace/VIRAT/annotations/" in video_path:
        video_path = video_path.replace( "/home/norm/workspace/VIRAT/annotations/", "/home/norm/workspace/VIRAT/videos/")

st.write("**Video Path:**", video_path)

st.sidebar.header("Options")
show_bbox = st.sidebar.checkbox("Show the bbox", value=False)
line_option = st.sidebar.radio("Line Style", ("Simple", "Arrowhead"))
selected_frame = st.sidebar.slider(
    "Select Frame",
    selected_event["start_frame"],
    selected_event["end_frame"],
    selected_event["start_frame"]
)
arrowhead = (line_option == "Arrowhead")
show_prompt = st.sidebar.checkbox("Show Prompt", value=False)



background = crop_video(video_path, selected_frame)  # screenshot at selected frame
if background is None:
    st.error(f"Could not retrieve the image from the video at frame {selected_frame}.")
    st.stop()

background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
if show_bbox:
    background_rgb = draw_bbox(background_rgb, selected_event, selected_frame)

img_with_traj = visualize_trajectory_img(trajectories, background_rgb.copy(), arrowhead=arrowhead)
st.image(img_with_traj, caption=f"Frame {selected_frame}", use_column_width=True)

if show_prompt:
    activity = selected_event.get("activity", "")
    prompt_text = generate_prompt(activity, trajectories)
    st.markdown("#### Generated Prompt")
    st.text_area("Prompt", prompt_text, height=200)



cols = st.columns(2)
with cols[0]:
    st.button("KEEP", on_click=go_to_next_case)
with cols[1]:
    st.button("REMOVE", on_click=remove_case)
