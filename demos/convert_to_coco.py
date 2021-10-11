from absl import app
from absl import flags
from aist_plusplus.loader import AISTDataset

import os
import numpy as np
import cv2
import pickle
import random
import json
from multiprocessing import Process

LHIP_POS = 11
RHIP_POS = 12
NUM_CPUS = 32
FLAGS = flags.FLAGS
flags.DEFINE_string(
  'anno_dir',
  '/home/ubuntu/RawDatasets/aist/annotations', 
  'input local dictionary for AIST++ annotations.'
)
flags.DEFINE_string(
  'video_dir',
  '/home/ubuntu/RawDatasets/aist/videos',
  'input local dictionary for AIST Dance Videos.'
)
flags.DEFINE_string(
  'output_dir',
  '/home/ubuntu/ProcessedDatasets/aist_processed/',
  'output directory for AIST frames'
)

def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def add_pelvis(points):
  new_points = np.zeros((points.shape[0] + 1, points.shape[1]))
  new_points[0:17, :] = points
  pelv = (points[LHIP_POS] + points[RHIP_POS])/2.0
  new_points[17] = pelv
  return new_points

def world_to_cam(points, rvec, tvec):
  rmat, jac = cv2.Rodrigues(rvec)
  mat = np.array([
    [rmat[0][0], rmat[0][1], rmat[0][2], tvec[0]],
    [rmat[1][0], rmat[1][1], rmat[1][2], tvec[1]],
    [rmat[2][0], rmat[2][1], rmat[2][2], tvec[2]]
  ])
  points_expanded = np.vstack((points.T, np.ones(points.shape[0])))
  return np.dot(mat, points_expanded).T

def get_video_lists(anno_dir):
  all_txt_f = open(os.path.join(anno_dir, "splits/pose_train.txt"))
  lines = all_txt_f.readlines()
  all_txt_f.close()
  final_list = []
  for l in lines:
    seq_name, view = AISTDataset.get_seq_name(l)
    if (view == "cAll"):
      for view in AISTDataset.VIEWS:
        name_with_cam = l.replace("cAll", view)
        final_list.append(name_with_cam.strip())
    else:
      final_list.append(l.strip())

  return final_list

def get_camera(camera_group, name):
  for camera in camera_group.cameras:
    if (camera.name == name):
      return camera

def get_bbox(uv, frame_shape):
  x = min(uv[:, 0]) - 10
  y = min(uv[:, 1]) - 10

  x_max = min(max(uv[:, 0]) + 10, frame_shape[1])
  y_max = min(max(uv[:, 1]) + 10, frame_shape[0])

  return [
      float(max(0, x)), float(max(0, y)), float(x_max - x), float(y_max - y)
  ]

def process(video_list, max_num, machine_num):
  aist_dataset = AISTDataset(FLAGS.anno_dir)

  # This is to avoid any overlap in the ids
  total = max_num * machine_num
  output = {
    "images": [],
    "annotations": [],
    "categories": [{
      'supercategory': 'person',
      'id': 1,
      'name': 'person'
    }]
  }
  vid_list_num = 0
  for video_name in video_list:
    video_path = os.path.join(FLAGS.video_dir, f'{video_name}.mp4')
    seq_name, view = AISTDataset.get_seq_name(video_name)
    env_name = aist_dataset.mapping_seq2env[seq_name]
    view_idx = AISTDataset.VIEWS.index(view)
    cgroup = AISTDataset.load_camera_group(aist_dataset.camera_dir, env_name)
    camera = get_camera(cgroup, view)
    keypoints3d_world = AISTDataset.load_keypoint3d(
        aist_dataset.keypoint3d_dir, seq_name, use_optim=True)
    nframes, njoints, _ = keypoints3d_world.shape
    keypoints2d = cgroup.project(keypoints3d_world)
    keypoints2d = keypoints2d.reshape(9, nframes, njoints, 2)[view_idx]
    K = camera.get_camera_matrix()
    # Convert world coords to camera coords
    vid_path = os.path.join(FLAGS.output_dir, video_name)
    os.makedirs(vid_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      
      if (random.random() > .5):
        i += 1
        continue

      cv2.imwrite(f"{vid_path}/{i:06d}.jpg", frame)
      output["images"].append({
        "id": total,
        "width": frame.shape[1],
        "height": frame.shape[0],
        "file_name": f"{video_name}/{i:06d}.jpg",
        "camera_param": {
            "focal": [float(K[0][0]), float(K[1][1])],
            "princpt": [float(K[0][2]), float(K[1][2])]
        }
      })
      
      keypoints3d_camera = world_to_cam(
        keypoints3d_world[i],
        camera.rvec,
        camera.tvec
      )

      keypoints3d_camera_pelvis = add_pelvis(keypoints3d_camera)
      keypoints2d_pelvis = add_pelvis(keypoints2d[i])
      output["annotations"].append({
        "id": total,
        "image_id": total,
        "category_id": 1,
        "is_crowd": 0,
        "joint_cam": keypoints3d_camera_pelvis.tolist(),
        "bbox": get_bbox(keypoints2d_pelvis, frame.shape) # x, y, w, h
      })
      i += 1
      total += 1
    vid_list_num += 1
    print(f"Completed {vid_list_num} of {len(video_list)} on machine {machine_num}")
  f = open(f"aist_training_{machine_num}.json", "w")
  json.dump(output, f)
  f.close()

def get_max_frames(video_list):
  max_num = 0
  for video_name in video_list:
    video_path = os.path.join(FLAGS.video_dir, f'{video_name}.mp4')
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if (total >= max_num):
      max_num = total

  return max_num

def main(_):
  video_list = get_video_lists(FLAGS.anno_dir)
  max_frames = get_max_frames(video_list)
  print(f"Max frames: {max_frames}")
  partitioned_list = partition(video_list, NUM_CPUS)
  processes = []
  for i in range(NUM_CPUS):
    processes.append(Process(target=process, args=(partitioned_list[i], max_frames, i)))
  for p in processes:
    p.start()
  for p in processes:
    p.join()

if __name__ == "__main__":
    app.run(main)
