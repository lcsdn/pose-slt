import sys
import os
from typing import Optional, Union

import numpy as np
import cv2
import torch
from torchvision.transforms import ToTensor
from PIL import Image

_thisdir = os.path.realpath(os.path.dirname(__file__))
_parentdir = os.path.dirname(_thisdir)
sys.path.append(_parentdir)
sys.path.append(os.path.join(_parentdir, "dope", "lcrnet-v2-improved-ppi"))
from dope.model import dope_resnet50, num_joints
from dope import postprocess, visu
from lcr_net_ppi_improved import LCRNet_PPI_improved

class PoseEstimator:
    """Estimates body pose keypoints from images using a DOPE model.
    
    Args:
        model_path (str): File location of the pytorch checkpoint for the
            desired DOPE model.
        postprocessing (str): Postprocessing operation. "ppi" or "nms".
            Default: "ppi".
        half (bool): If True, runs the network with half precision. Default: True.
    """
    def __init__(self,
                 model_path: str,
                 postprocessing: str="ppi",
                 half: bool = True) -> None:
        assert os.path.isfile(model_path)
        assert postprocessing in ["nms", "ppi"]
        self.model_path = model_path
        self.postprocessing = postprocessing
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Load the model from file
        ckpt = torch.load(self.model_path, map_location=self.device)
        ckpt['dope_kwargs']['rpn_post_nms_top_n_test'] = 1000
        self.half = half and ckpt['half']
        self.model = dope_resnet50(**ckpt['dope_kwargs'])
        if self.half:
            self.model = self.model.half()
        self.model = self.model.eval()
        self.model.load_state_dict(ckpt['state_dict'])
        self.model = self.model.to(self.device)
        self.ppi_kwargs = {part+'_ppi_kwargs': ckpt[part+'_ppi_kwargs']
                           for part in ["body", "hand", "face"]}

    def load_image(self, image: Union[Image.Image, str]) -> list:
        # Load the image if given file path
        if isinstance(image, str):
            assert os.path.isfile(image)
            image_path = image
            image = Image.open(image)
        #image = image.resize((210, 300)) # compensate distorsion of images from PHOENIX-14T
        imlist = [ToTensor()(image).to(self.device)]
        if self.half:
            imlist = [im.half() for im in imlist]
        return imlist

    def detect_keypoints(self,
               image: Union[Image.Image, str],
               trim: bool = False,
               visu: bool = False,
               postprocessing: Optional[str] = None) -> dict:
        if postprocessing is None:
            postprocessing = self.postprocessing
        assert postprocessing in ["nms", "ppi"]
        
        imlist = self.load_image(image)
        resolution = imlist[0].size()[-2:]
        
        # Forward pass
        with torch.no_grad():
            results = self.model(imlist, None)[0]
            
        # Postprocess the results
        # (pose proposals integration, wrists/head assignment)
        detections = self._postprocess_results(results, resolution, postprocessing)
        
        if trim:
            detections['body'] = detections['body'][:1]
            detections['face'] = detections['face'][:1]
            detections['hand'] = detections['hand'][:2]
        
        if visu:
            assert isinstance(image, str)
            image_path = image
            assert os.path.isfile(image_path)
            image = Image.open(image)
            output_path = f"{image_path}_{os.path.basename(self.model_path)[:-8]}_{postprocessing}.jpg"
            self._visualisation_2d(detections, image, output_path)
    
        return detections
        
    def compute_features(self, image: Union[Image.Image, str]) -> torch.tensor:
        imlist= self.load_image(image)
        with torch.no_grad():
            images, targets = self.model.transform(imlist)
            features = self.model.backbone(images.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([('0', features)])
            proposals, _ = self.model.rpn(images, features, targets)
        return proposals

    def _postprocess_results(self,
                             results: dict,
                             resolution: torch.Size,
                             postprocessing: str) -> dict:
        parts = ["body", "hand", "face"]
        if postprocessing == "ppi":
            res = {k: v.float().data.cpu().numpy() for k, v in results.items()}
            detections = {}
            for part in parts:
                detections[part] = LCRNet_PPI_improved(res[part+'_scores'],
                                                       res['boxes'],
                                                       res[part+'_pose2d'],
                                                       res[part+'_pose3d'],
                                                       resolution,
                                                       **self.ppi_kwargs[part+'_ppi_kwargs'])
        else:
            detections = {}
            for part in parts:
                dets, indices, bestcls = postprocess.DOPE_NMS(results[part+'_scores'],
                                                              results['boxes'],
                                                              results[part+'_pose2d'],
                                                              results[part+'_pose3d'],
                                                              min_score=0.)
                dets = {k: v.float().data.cpu().numpy() for k,v in dets.items()}
                detections[part] = [{'score': dets['score'][i],
                                     'pose2d': dets['pose2d'][i,...],
                                     'pose3d': dets['pose3d'][i,...]} for i in range(dets['score'].size)]
                if part == "hand":
                    for i in range(len(detections[part])):
                        detections[part][i]['hand_isright'] = bestcls < self.ppi_kwargs['hand_ppi_kwargs']['K']
                        
        # Assignment of hands and head to body
        detections, _, _ = postprocess.assign_hands_and_head_to_body(detections)

        return detections
    
    @staticmethod
    def _visualisation_2d(detections: dict, image: Image.Image, output_path: str) -> None:
        """Save a visualisation of the detected keypoints at output_path."""
        det_poses2d = {part: np.stack([d['pose2d'] for d in part_detections], axis=0) \
                            if len(part_detections)>0 \
                            else np.empty((0, num_joints[part], 2), dtype=np.float32) \
                            for part, part_detections in detections.items()}
        scores = {part: [d['score'] \
                        for d in part_detections] \
                        for part, part_detections in detections.items()}
        imout = visu.visualize_bodyhandface2d(np.asarray(image)[:, :, ::-1],
                                            det_poses2d,
                                            dict_scores=scores)
        cv2.imwrite(output_path, imout)


def assign_hands(detections: dict) -> None:
    """Assigns a position (0: left, 1: right) to detected hands."""
    # If one hand is detected, use position in the image
    if len(detections["hand"]) == 1:
        isright = detections["hand"][0]["pose2d"][:, 0].mean() >= 105
        detections["hand"][0]["position"] = int(isright)
    # If two hands are detected, use relative position
    elif len(detections["hand"]) == 2:
        first_isright = (detections["hand"][0]["pose2d"][:, 0]
                         - detections["hand"][1]["pose2d"][:, 0]).mean() >= 0
        detections["hand"][0]["position"] = int(first_isright)
        detections["hand"][1]["position"] = int(not first_isright)


def sequence_to_keypoints(sequence_path: str,
                              pose_estimator: PoseEstimator) -> dict:
    """Computes a dictionary of body keypoints in torch.tensor format
    for a given sequence path in the PHOENIX14T dataset."""
    sequence_basepath, extension = os.path.splitext(sequence_path)
    is_video = extension == ".mp4"

    if is_video:
        cap = cv2.VideoCapture(sequence_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        n_frames = len(os.listdir(sequence_path))

    sequence_keypoints = {"name": os.path.basename(sequence_basepath),
                          "body_score": torch.zeros(n_frames, dtype=torch.float32),
                          "body_keypoints": torch.zeros((n_frames, 13, 5), dtype=torch.float32),
                          "hand_score": torch.zeros((n_frames, 2), dtype=torch.float32),
                          "hand_keypoints": torch.zeros((n_frames, 2, 21, 5), dtype=torch.float32),
                          "face_score": torch.zeros(n_frames, dtype=torch.float32),
                          "face_keypoints": torch.zeros((n_frames, 84, 5), dtype=torch.float32)}

    for i_frame in range(n_frames):
        if is_video:
            _, array_image = cap.read()
            image = Image.fromarray(cv2.cvtColor(array_image, cv2.COLOR_BGR2RGB))
        else:
            filename = "images%04d.png" % (i_frame+1)
            image = os.path.join(sequence_path, filename)
            assert os.path.isfile(image)
        
        detections = pose_estimator(image, trim=True)
        assign_hands(detections)

        for part, list_detections in detections.items():
            for part_detection in list_detections:
                if part == "hand":
                    loc = (i_frame, part_detection["position"])
                else:
                    loc = i_frame
                sequence_keypoints[part+"_score"][loc] = torch.tensor(part_detection["score"])
                keypoints = sequence_keypoints[part+"_keypoints"][loc]
                keypoints[:, :2] = torch.from_numpy(part_detection["pose2d"])
                keypoints[:, 2:] = torch.from_numpy(part_detection["pose3d"])

    return sequence_keypoints


def build_keypoints_set(dataset_path: str, pose_estimator: PoseEstimator, verbose=False) -> list:
    """Take the path to the PHOENIX14T dataset and return a list containing the
    keypoints dictionary of each subset (train, dev, test).
    
    The keys of these dictionaries are the sequences in the subset, and the values
    are the keypoints of each sequence extracted by pose_estimator."""
    keypoints_set = []
    for subset_name in ["train", "dev", "test"]:
        subset_path = os.path.join(dataset_path, subset_name)
        if os.path.isdir(subset_path):
            if verbose:
                print(f"Found {subset_name} set")
                i = 0
            sequence_filenames = os.listdir(subset_path)
            keypoints_subset = {}
            for filename in sequence_filenames:
                keypoints = sequence_to_keypoints(
                    os.path.join(subset_path, filename),
                    pose_estimator
                )
                keypoints["name"] = "{}/{}".format(subset_name, keypoints["name"])
                keypoints_subset[keypoints["name"]] = keypoints
                if verbose:
                    i += 1
                    print(f'({int(100 * i / len(sequence_filenames)):02}%) Extracted keypoints in sequence "{subset_name}/{filename}"')
            keypoints_set.append(keypoints_subset)
        else:
            if verbose:
                print(f"Cannot find {subset_name} set")
            keypoints_set.append(None)
    return keypoints_set