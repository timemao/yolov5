# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path
import struct

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

is_debug=0
#is_debug=1

is_save_anchor=0
#is_save_anchor=1

def raw2arr(raw_path,n):
    f = open(raw_path, 'rb')
    arr = struct.unpack('f' * n, f.read(4 * n))
    arr=np.array(arr)
    f.close()
    return arr

def reshape_tensor(out,h,w,n_input=18):
    x=np.reshape(out,[h*w,-1]).T  # 18x(hxw)
    y=np.reshape(x,[n_input,h,-1])
    return np.expand_dims(y,axis=0)

is_replace_x=0
#is_replace_x=1

n_classes = 2
n_input = (5 + n_classes) * 3
n_h, n_w = 92, 120

is_export_snpe=1
#is_export_snpe=0

if (is_replace_x):
    path_raw1 = r"D:\data\data_tuoming2\res_quant\Result_1\onnx__Reshape_386.raw"
    path_raw2 = r"D:\train_result\face_detection\dlc\268.raw"
    path_raw3 = r"D:\train_result\face_detection\dlc\270.raw"

    path_raw1 = r"D:\data\data_tuoming2\res_quant\Result_1\onnx__Reshape_386.raw"
    path_raw2 = r"D:\data\data_tuoming2\res_quant\Result_1\onnx__Reshape_424.raw"
    path_raw3 = r"D:\data\data_tuoming2\res_quant\Result_1\onnx__Reshape_462.raw"

    out1 = raw2arr(path_raw1, n_input * n_h * n_w)
    out2 = raw2arr(path_raw2, n_input * n_h//2 * n_w//2)
    out3 = raw2arr(path_raw3, n_input * n_h//4*n_w//4)

    x1 = torch.tensor(reshape_tensor(out1, n_h,n_w,n_input=n_input))
    x2 = torch.tensor(reshape_tensor(out2, n_h//2,n_w//2,n_input=n_input))
    x3 = torch.tensor(reshape_tensor(out3, n_h//4,n_w//4,n_input=n_input))

def write_tensor(tensor,fw):
    n, c, h, w = tensor.shape
    fw.write("shape=[%d,%d,%d,%d]\n"%(n,c,h,w))
    c1, h1, w1 = min(c, 1), min(h, 10), min(w, 10)
    for k in range(c1):
        fw.write("c=%d\n" % k)
        for i in range(h1):
            show_str = ""
            for j in range(w1):
                show_str += "%.06f " % tensor[0, k, i, j]
            fw.write(show_str + "\n")

class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output

        if(is_debug):
            fw=open("Detect_conv.txt","w")

        for i in range(self.nl):
            if(is_debug):
                print("IDetect.forward input ", x[i].shape)
                fw.write("#-------------------layer_index=%d--------------------#\n"%i)
                write_tensor(x[i],fw)

            grid_txt=r"D:\train_result\yolov5\train_5s_960_order\grid%d.txt"%i
            anchor_txt=r"D:\train_result\yolov5\train_5s_960_order\anchor_grid%d.txt" % i
            if(is_save_anchor and not os.path.exists(grid_txt) and not os.path.exists(anchor_txt)):
                model_name="yolov5"
                fw_grid=open(grid_txt,"w")
                fw_anchor=open(anchor_txt,"w")

            x[i] = self.m[i](x[i])  # conv

            if (is_replace_x):
                if (i == 0):
                    x[i] = x1
                elif (i == 1):
                    x[i] = x2
                elif (i == 2):
                    x[i] = x3

            if (is_debug):
                print("IDetect.forward output ",i, x[i].shape)
                write_tensor(x[i], fw)

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            if(is_export_snpe):
                x[i] = x[i].view(self.na, self.no, ny, nx).permute(0, 2, 3, 1).contiguous()
            else:
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if(is_debug):
                xx=x[i].squeeze(0)
                xx=xx.view(self.na,-1,self.no).unsqueeze(0)
                write_tensor(xx,fw)

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if(is_save_anchor and not os.path.exists(grid_txt) and not os.path.exists(anchor_txt)):
                    print(i, self.grid[i].shape, self.anchor_grid[i].shape)
                    tmp = self.grid[i].squeeze()  # 12x20x2
                    tmp2 = self.anchor_grid[i].squeeze()  # 3x2

                    s1,s2,s3,_=tmp.shape
                    for tmpi in range(s1):
                        for tmpj in range(s2):
                            for k in range(s3):
                                fw_grid.write("%f,%f\n"%(tmp[tmpi,tmpj,k,0],tmp[tmpi,tmpj,k,1]))
                                fw_anchor.write("%f,%f\n"%(tmp2[tmpi,tmpj,k,0],tmp2[tmpi,tmpj,k,1]))

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)

                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    if(is_export_snpe):
                        xy,wh,conf=x[i].sigmoid().view(bs,self.na,ny,nx,self.no).split((2, 2, self.nc + 1), 4)
                    else:
                        xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

            if(is_save_anchor and not os.path.exists(grid_txt) and not os.path.exists(anchor_txt)):
                fw_grid.close()
                fw_anchor.close()

        if(is_debug):
            for i in range(3):
                print("self.anchor_grid=", self.anchor_grid[i][0,:,0,0,:])
            fw.close()

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
        #return x

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    # YOLOv5 base model
    def get_conv_macc(self,op,in_w,in_h,out_w=None,out_h=None):
        in_channels = op.in_channels
        out_channels = op.out_channels
        kernel_size = op.kernel_size[0]
        padding = op.padding
        stride = op.stride

        out_w=in_w//stride[0]
        out_h=in_h//stride[1]

        macc=(kernel_size * kernel_size * in_channels) * (out_w * out_h) * out_channels
        macc=macc
        param=kernel_size * kernel_size * in_channels * out_channels

        return int(macc/1000/1000),int(param/1000),out_w,out_h

    def get_C3_macc(self,op,in_w,in_h):
        # 左边分支
        macc1_1,param1_1,out_w,out_h=self.get_conv_macc(op.cv1.conv,in_w=in_w,in_h=in_h)
        macc1,param1=macc1_1,param1_1
        n=len(op.m)
        for i in range(n):
            macc1_2,param1_2,out_w,out_h=self.get_conv_macc(op.m[i].cv1.conv,in_w=out_w,in_h=out_h)
            macc1_3,param1_3,out_w,out_h=self.get_conv_macc(op.m[i].cv2.conv,in_w=out_w,in_h=out_h)

            macc1+=macc1_2+macc1_3
            param1+=param1_2+param1_3
        # 右边分支
        macc2, param2, out_w, out_h = self.get_conv_macc(op.cv1.conv, in_w, in_h)

        # 后接卷积
        macc3, param3, _, _ = self.get_conv_macc(op.cv3.conv,in_w=out_w,in_h=out_h)

        return macc1+macc2+macc3, param1+param2+param3

    def get_SPPF_macc(self,op,in_w,in_h):
        macc1, param1, out_w, out_h = self.get_conv_macc(op.cv1.conv,in_w,in_h)
        macc2, param2,out_w,out_h = self.get_conv_macc(op.cv2.conv,in_w=out_w,in_h=out_h)

        return macc1+macc2, param1+param2

    def get_Detect_macc(self,op,x_list):
        macc1, param1, out_w, out_h = self.get_conv_macc(op.m[0],x_list[0].shape[2],x_list[0].shape[3])
        macc2, param2, out_w, out_h = self.get_conv_macc(op.m[1],x_list[1].shape[2],x_list[1].shape[3])
        macc3, param3, out_w, out_h = self.get_conv_macc(op.m[2],x_list[2].shape[2],x_list[2].shape[3])

        return macc1+macc2+macc3, param1+param2+param3

    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        y1=[x]
        t1=['input']
        f1=[-1]
        idx_layer=0

        macc_list=[]
        params_list=[]

        macc_sum=0
        param_sum=0

        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)

            if(is_debug):
                print(m.type)
                if(m.type=="models.yolo.IDetect"):
                    a=0

            print("#------------%d, %s------------#"%(idx_layer,m.type))
            #print(m)
            if("Conv" in m.type):
                #print("Conv out.shape=",x.shape)
                macc,param,_,_=self.get_conv_macc(m.conv,x.shape[2],x.shape[3])
                macc_list.append(macc)
                params_list.append(param)
                macc_sum+=macc
                param_sum+=param
                print("macc,param= %d Mb, %d Kb" % (macc, param))
            elif("C3" in m.type):
                #print("C3")
                macc, param = self.get_C3_macc(m,x.shape[2],x.shape[3])
                macc_list.append(macc)
                params_list.append(param)
                macc_sum+=macc
                param_sum+=param
                print("macc,param= %d Mb, %d Kb" % (macc, param))
            elif("SPPF" in m.type):
                #print("SPPF")
                macc, param = self.get_SPPF_macc(m,x.shape[2],x.shape[3])
                macc_list.append(macc)
                params_list.append(param)
                macc_sum+=macc
                param_sum+=param
                print("macc,param= %d Mb, %d Kb" % (macc, param))
            elif("Detect" in m.type):
                #print("Detect")
                macc, param = self.get_Detect_macc(m,x)
                macc_list.append(macc)
                params_list.append(param)
                macc_sum+=macc
                param_sum+=param
                print("macc,param= %d Mb, %d Kb" % (macc, param))
            else:
                a=0
                #print("!!! lost compute param cost !!!")

            idx_layer+=1

            x = m(x)  # run

            y1.append(x)
            t1.append(m.type)
            f1.append(m.f)

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

        print("#--------- model macc, param =%d Mb, %d Kb -------------#\n\n\n\n\n"%(macc_sum,param_sum))

        if(is_debug):
            return x, y1, t1, f1
        else:
            return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None


def parse_model(d, ch):  # model_dict, input_channels(3)
    # Parse a YOLOv5 model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
                Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
