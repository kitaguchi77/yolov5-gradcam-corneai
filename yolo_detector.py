
import torch
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage
from deep_utils.utils.box_utils.boxes import Box
from utils.general import non_max_suppression as yolo_nms, xywh2xyxy
from utils.datasets import letterbox
from models.experimental import attempt_load
from models.yolo import Model, Detect
from models.common import Conv, Bottleneck, C3, SPPF, Concat
from torch.nn import Sequential, Conv2d, BatchNorm2d, SiLU, MaxPool2d, Upsample, ModuleList

torch.serialization.add_safe_globals([
    Model, Detect, Sequential, ModuleList, Conv, Bottleneck, C3, SPPF, Concat,
    Conv2d, BatchNorm2d, SiLU, MaxPool2d, Upsample
])

class YOLOV5TorchObjectDetector(torch.nn.Module):
    def __init__(self,
                 model_weight,
                 device,
                 img_size,
                 names=None,
                 mode="eval",
                 confidence=0.25,
                 iou_thresh=0.45,
                 agnostic_nms=False):
        super(YOLOV5TorchObjectDetector, self).__init__()
        self.device = device
        self.model = None
        self.img_size = img_size
        self.mode = mode
        self.confidence = confidence
        self.iou_thresh = iou_thresh
        self.agnostic = agnostic_nms

        self.model = attempt_load(model_weight, device=device)
        print("[INFO] Model is loaded")

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(device)

        if self.mode == "train":
            self.model.train()
        else:
            self.model.eval()

        self.names = names

    @staticmethod
    def non_max_suppression(prediction, logits, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                            multi_label=False, labels=(), max_det=300):
        import time
        nc = prediction.shape[2] - 5
        xc = prediction[..., 4] > conf_thres

        assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}"
        assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}"

        min_wh, max_wh = 2, 4096
        max_nms = 30000
        time_limit = 10.0
        redundant = True
        multi_label &= nc > 1
        merge = False

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        logits_output = [torch.zeros((0, 80), device=logits.device)] * logits.shape[0]

        for xi, (x, log_) in enumerate(zip(prediction, logits)):
            x = x[xc[xi]]
            log_ = log_[xc[xi]]

            if not x.shape[0]:
                continue

            x[:, 5:] *= x[:, 4:5]
            box = xywh2xyxy(x[:, :4])

            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            log_ = log_[conf.view(-1) > conf_thres]

            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            n = x.shape[0]
            if not n:
                continue
            elif n > max_nms:
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]

            c = x[:, 5:6] * (0 if agnostic else max_wh)
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torch.ops.torchvision.nms(boxes, scores, iou_thres)

            if i.shape[0] > max_det:
                i = i[:max_det]

            output[xi] = x[i]
            logits_output[xi] = log_[i]

        return output, logits_output

    @staticmethod
    def yolo_resize(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        return letterbox(img, new_shape=new_shape, color=color, auto=auto, scaleFill=scaleFill, scaleup=scaleup)

    def forward(self, img):
        prediction, logits, _ = self.model(img, augment=False)
        prediction, logits = self.non_max_suppression(
            prediction, logits, self.confidence, self.iou_thresh,
            classes=None, agnostic=self.agnostic
        )

        self.boxes, self.class_names, self.classes, self.confidences = [[[] for _ in range(img.shape[0])] for _ in range(4)]

        for i, det in enumerate(prediction):
            if len(det):
                for *xyxy_tensor, conf, cls in det:
                    # Convert tensor elements to Python floats, then to integers
                    x1, y1, x2, y2 = float(xyxy_tensor[0]), float(xyxy_tensor[1]), float(xyxy_tensor[2]), float(xyxy_tensor[3])

                    # Clamp xyxy values to image size
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(self.img_size[0], int(x2)) # self.img_size[0] is width
                    y2 = min(self.img_size[1], int(y2)) # self.img_size[1] is height

                    bbox = [x1, y1, x2, y2] # Directly create the list of integers
                    self.boxes[i].append(bbox)
                    self.confidences[i].append(round(conf.item(), 2))
                    cls = int(cls.item())
                    self.classes[i].append(cls)
                    if self.names is not None:
                        self.class_names[i].append(self.names[cls])
                    else:
                        self.class_names[i].append(cls)
        return [self.boxes, self.classes, self.class_names, self.confidences], logits

    def preprocessing(self, img):
        if len(img.shape) != 4:
            img = np.expand_dims(img, axis=0)
        im0 = img.astype(np.uint8)

        resized_imgs = [self.yolo_resize(im, new_shape=self.img_size)[0] for im in im0]
        img = np.array(resized_imgs)
        img = img.transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img / 255.0
        return img
