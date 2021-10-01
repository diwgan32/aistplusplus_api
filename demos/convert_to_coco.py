from absl import app
from absl import flags
from aist_plusplus.loader import AISTDataset

import numpy as np
import cv2
import pickle
import random
import json

FLAGS = flags.FLAGS
flags.DEFINE_string(
  'anno_dir',
  '/usr/local/google/home/ruilongli/data/public/aist_plusplus_final/',
  'input local dictionary for AIST++ annotations.'
)
flags.DEFINE_string(
  'video_dir',
  '/usr/local/google/home/ruilongli/data/AIST_plusplus/refined_10M_all_video/',
  'input local dictionary for AIST Dance Videos.'
)
flags.DEFINE_string(
  'output_dir',
  '/home/ubuntu/ProcessedDatasets/aist/',
  'output directory for AIST frames'
)

def get_video_lists(anno_dir):
  all_txt_f = open(os.path.join(anno_dir, "splits/all.txt"))
  lines = all_txt_f.readlines()
  all_txt_f.close()
  return lines

def get_camera(camera_group, name):
  for camera in camera_group.cameras:
    if (camera.name == name):
      return camera

def get_bbox(uv, frame_shape):
  x = min(uv[:, 0]) - 10
  y = min(uv[:, 1]) - 10

  x_max = min(max(uv[:, 0]) + 10, frame_shape.shape[1])
  y_max = min(max(uv[:, 1]) + 10, frame_shape.shape[0])

  return [
      float(max(0, x)), float(max(0, y)), float(x_max - x), float(y_max - y)
  ]

if __name__ == "__main__":
  video_list = get_video_lists(FLAGS.anno_dir)
  aist_dataset = AISTDataset(FLAGS.anno_dir)
  total = 0
  output = {
    "images": [],
    "annotations": [],
    "categories": [{
      'supercategory': 'person',
      'id': 1,
      'name': 'person'
    }]
  }

  for video_name in video_list:
    video_path = os.path.join(FLAGS.video_dir, f'{video_name}.mp4')
    seq_name, view = AISTDataset.get_seq_name(video_name)
    view_idx = AISTDataset.VIEWS.index(view)

    keypoints3d_world = AISTDataset.load_keypoint3d(
        aist_dataset.keypoint3d_dir, seq_name, use_optim=True)
    nframes, njoints, _ = keypoints3d_world.shape
    keypoints2d = cgroup.project(keypoints3d_world)
    keypoints2d = keypoints2d.reshape(9, nframes, njoints, 2)[view_idx]
    env_name = aist_dataset.mapping_seq2env[seq_name]
    cgroup = AISTDataset.load_camera_group(aist_dataset.camera_dir, env_name)
    camera = get_camera(cgroup, view) # TODO implement
    K = camera.get_camera_matrix()
    # Convert world coords to camera coords
    keypoints_3d_camera = cv2.projectPoints(
      keypoints_3d,
      camera.rvec,
      camera.tvec,
      np.eye(3),
      np.zeros(3)
    )
    vid_path = os.path.join(FLAGS.output_dir, video_name)
    os.makedirs(vid_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
      ret, frame = cap.read()
      if not ret:
        break

      cv2.imwrite(frame, f"{vid_path}/{i:06d}.jpg")
      output["images"].append({
        "id": total,
        "width": frame.shape[1],
        "height": frame.shape[2],
        "file_name": f"{vid_path}/{i:06d}.jpg",
        "camera_param": {
            "focal": [float(K[0][0]), float(K[1][1])],
            "princpt": [float(K[0][2]), float(K[1][2])]
        }
      })

      output["annotations"].append({
        "id": total,
        "image_id": total,
        "category_id": 1,
        "is_crowd": 0,
        "joint_cam": keypoints_3d_camera[i].tolist(),
        "bbox": get_bbox(keypoints2d, frame.shape) # x, y, w, h
      })

      i += 1
      total += 1


    if (total > 3000):
      break

  f = open("aist_training.json", "w")
  json.dump(output, f)
  f.close()