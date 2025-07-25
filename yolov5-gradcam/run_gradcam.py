

import argparse
import os
import cv2
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
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

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

            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]
                v[:, 4] = 1.0
                v[range(len(l)), l[:, 0].long() + 5] = 1.0
                x = torch.cat((x, v), 0)

            if not x.shape[0]:
                continue

            x[:, 5:] *= x[:, 4:5]
            box = xywh2xyxy(x[:, :4])

            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:
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

            if merge and (1 < n < 3E3):
                iou = box_iou(boxes[i], boxes) > iou_thres
                weights = iou * scores[None]
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
                if redundant:
                    i = i[iou.sum(1) > 1]

            output[xi] = x[i]
            logits_output[xi] = log_[i]
            assert log_[i].shape[0] == x[i].shape[0]

            if (time.time() - t) > time_limit:
                print(f"WARNING: NMS time limit {time_limit}s exceeded")
                break

        return output, logits_output

    @staticmethod
    def yolo_resize(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        return letterbox(img, new_shape=new_shape, color=color, auto=auto, scaleFill=scaleFill, scaleup=scaleup)

    def forward(self, img):
        with torch.no_grad():
            prediction, logits, _ = self.model(img, augment=False)
            prediction, logits = self.non_max_suppression(
                prediction, logits, self.confidence, self.iou_thresh,
                classes=None, agnostic=self.agnostic
            )

            self.boxes, self.class_names, self.classes, self.confidences = [[[] for _ in range(img.shape[0])] for _ in range(4)]

            for i, det in enumerate(prediction):
                if len(det):
                    det_cpu = det.cpu()
                    del det

                    for *xyxy, conf, cls in det_cpu:
                        xyxy[0] = max(0, xyxy[0])
                        xyxy[1] = max(0, xyxy[1])
                        xyxy[2] = min(self.img_size[0], xyxy[2])
                        xyxy[3] = min(self.img_size[1], xyxy[3])

                        bbox = Box.box2box(xyxy,
                                           in_source=Box.BoxSource.Torch,
                                           to_source=Box.BoxSource.Numpy,
                                           return_int=True)
                        self.boxes[i].append(bbox)
                        self.confidences[i].append(round(conf.item(), 2))
                        cls = int(cls.item())
                        self.classes[i].append(cls)
                        if self.names is not None:
                            self.class_names[i].append(self.names[cls])
                        else:
                            self.class_names[i].append(cls)

            del prediction

        return [self.boxes, self.classes, self.class_names, self.confidences], logits

    def preprocessing(self, img):
        if len(img.shape) != 4:
            img = np.expand_dims(img, axis=0)
        im0 = img.astype(np.uint8)
        del img

        resized_imgs = []
        for im in im0:
            resized_img = self.yolo_resize(im, new_shape=self.img_size)[0]
            resized_imgs.append(resized_img)

        img = np.array(resized_imgs)
        del resized_imgs, im0

        img = img.transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img / 255.0
        return img

def find_yolo_layer(model, layer_name):
   hierarchy = layer_name.split("_")
   target_layer = model.model._modules[hierarchy[0]]

   for h in hierarchy[1:]:
       target_layer = target_layer._modules[h]
   return target_layer

class YOLOV5GradCAM:
    def __init__(self, model, layer_name, img_size=(640, 640), method="gradcam", debug=False):
        self.model = model
        self.gradients = dict()
        self.activations = dict()
        self.method = method
        self.cls_names = []
        self.debug = debug
        self.layer_name = layer_name

        self.forward_handle = None
        self.backward_handle = None
        self.target_layer = None

        def backward_hook(module, grad_input, grad_output):
            try:
                self.gradients["value"] = grad_output[0].detach().clone()
                if self.debug:
                    print(f"[DEBUG] Successfully detached and cloned grad_output for {layer_name}")
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Error in backward_hook: {e}")
                raise
            return None

        def forward_hook(module, input, output):
            self.activations["value"] = output.detach().clone()
            return None

        self.target_layer = find_yolo_layer(self.model, layer_name)

        for param in self.model.model.parameters():
            param.requires_grad = False
        for param in self.target_layer.parameters():
            param.requires_grad = True

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_backward_hook(backward_hook)

    def cleanup(self):
        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None
        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None
        self.gradients.clear()
        self.activations.clear()
        self.target_layer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def __del__(self):
        try:
            self.cleanup()
        except:
            pass

    def forward(self, input_img, class_idx=True):
        saliency_maps = []
        b, c, h, w = input_img.size()

        preds, logits = self.model(input_img)

        with torch.no_grad():
            _, top3_indices = torch.topk(logits[0], k=3)

            if top3_indices.numel() > 0:
                preds[1][0] = top3_indices.tolist()[0]
                preds[2][0] = [self.model.names[i] for i in preds[1][0]]
                self.cls_names = preds[2][0]
            else:
                self.cls_names = []

        for cls, cls_name in zip(preds[1][0], preds[2][0]):
            self.model.zero_grad()
            input_img_clone = input_img.clone()
            _, new_logits = self.model(input_img_clone)
            del input_img_clone

            if class_idx:
                score = new_logits[0][0][cls]
            else:
                score = new_logits[0][0].max()

            score.backward()

            if "value" not in self.gradients or "value" not in self.activations:
                continue

            gradients = self.gradients["value"]
            activations = self.activations["value"]
            b, k, u, v = gradients.size()

            if self.method == "gradcam":
                weights = self._gradcam_weights(gradients, b, k)
            elif self.method == "gradcampp":
                weights = self._gradcampp_weights(gradients, activations, score, b, k, u, v)

            del score, new_logits

            with torch.no_grad():
                saliency_map = (weights * activations).sum(1, keepdim=True)
                del weights
                saliency_map = F.relu(saliency_map)
                saliency_map = F.interpolate(saliency_map, size=(h, w), mode="bilinear", align_corners=False)
                saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
                saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min + 1e-8).data
                del saliency_map_min, saliency_map_max

            saliency_maps.append(saliency_map)

            if "value" in self.gradients:
                del self.gradients["value"]
            if "value" in self.activations:
                del self.activations["value"]
            del gradients, activations

        return saliency_maps, logits, preds, self.cls_names

    def _gradcam_weights(self, gradients, b, k):
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        del alpha
        return weights

    def _gradcampp_weights(self, gradients, activations, score, b, k, u, v):
        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
            activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num.div(alpha_denom + 1e-7)
        del alpha_num, alpha_denom

        relu_grad = F.relu(score.exp() * gradients)
        weights = (relu_grad * alpha).view(b, k, u*v).sum(-1).view(b, k, 1, 1)
        del relu_grad, alpha

        return weights

    def __call__(self, input_img):
        return self.forward(input_img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the YOLOv5 model')
    parser.add_argument('--img-path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save the output')
    parser.add_argument('--layer-name', type=str, default='model_23_cv3_conv', help='The layer to visualize')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--method', type=str, default='gradcampp', choices=['gradcam', 'gradcampp'], help='Visualization method')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    
    names = ["infection","normal","non-infection","scar","tumor","deposit","APAC","lens opacity","bullous"]
    
    model = YOLOV5TorchObjectDetector(args.model_path, device, img_size=(args.img_size, args.img_size), names=names)

    img = cv2.imread(args.img_path)
    torch_img = model.preprocessing(img[..., ::-1])

    with YOLOV5GradCAM(model=model, layer_name=args.layer_name, img_size=(args.img_size, args.img_size), method=args.method) as saliency_method:
        masks, logits, [boxes, classes, class_names, confidences], cls_names = saliency_method(torch_img)

        for i, (mask, cls_name) in enumerate(zip(masks, cls_names)):
            # Convert mask to heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze().cpu()), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Superimpose the heatmap on original image
            result = (heatmap * 0.5 + img * 0.5).astype(np.uint8)

            # Save the result
            output_path = os.path.join(args.output_dir, f'{os.path.basename(args.img_path)}_{cls_name}_{args.layer_name}.jpg')
            cv2.imwrite(output_path, result)
            print(f"Saved to {output_path}")

if __name__ == '__main__':
    main()
