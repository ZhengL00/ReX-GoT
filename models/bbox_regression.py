
import torch
_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


class Box2BoxTransform(object):


    def __init__(self, weights, scale_clamp=_DEFAULT_SCALE_CLAMP):

        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes, target_boxes):

        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights

        target_widths = target_boxes[:, 2] - target_boxes[:, 0]
        target_heights = target_boxes[:, 3] - target_boxes[:, 1]
        target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights

        wx, wy, ww, wh = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)

        deltas = torch.stack((dx, dy, dw, dh), dim=1)
        assert (
            (src_widths > 0).all().item()
        ), "Input boxes to Box2BoxTransform are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):

        if not torch.isfinite(deltas).all().item():
            print(deltas)
            input('enter')
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
        return pred_boxes


class Box2BoxTransformRotated(object):

    def __init__(self, weights, scale_clamp=_DEFAULT_SCALE_CLAMP):

        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes, target_boxes):

        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        src_ctr_x, src_ctr_y, src_widths, src_heights, src_angles = torch.unbind(
            src_boxes, dim=1
        )

        (
            target_ctr_x,
            target_ctr_y,
            target_widths,
            target_heights,
            target_angles,
        ) = torch.unbind(target_boxes, dim=1)

        wx, wy, ww, wh, wa = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)
        da = target_angles - src_angles
        while len(torch.where(da < -180.0)[0]) > 0:
            da[torch.where(da < -180.0)] += 360.0
        while len(torch.where(da > 180.0)[0]) > 0:
            da[torch.where(da > 180.0)] -= 360.0
        da *= wa * math.pi / 180.0

        deltas = torch.stack((dx, dy, dw, dh, da), dim=1)
        assert (
            (src_widths > 0).all().item()
        ), "Input boxes to Box2BoxTransformRotated are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):

        assert deltas.shape[1] == 5 and boxes.shape[1] == 5
        assert torch.isfinite(deltas).all().item()

        boxes = boxes.to(deltas.dtype)

        ctr_x, ctr_y, widths, heights, angles = torch.unbind(boxes, dim=1)
        wx, wy, ww, wh, wa = self.weights
        dx, dy, dw, dh, da = torch.unbind(deltas, dim=1)

        dx.div_(wx)
        dy.div_(wy)
        dw.div_(ww)
        dh.div_(wh)
        da.div_(wa)

        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0] = dx * widths + ctr_x
        pred_boxes[:, 1] = dy * heights + ctr_y
        pred_boxes[:, 2] = torch.exp(dw) * widths
        pred_boxes[:, 3] = torch.exp(dh) * heights

        pred_angle = da * 180.0 / math.pi + angles

        while len(torch.where(pred_angle < -180.0)[0]) > 0:
            pred_angle[torch.where(pred_angle < -180.0)] += 360.0
        while len(torch.where(pred_angle > 180.0)[0]) > 0:
            pred_angle[torch.where(pred_angle > 180.0)] -= 360.0

        pred_boxes[:, 4] = pred_angle

        return pred_boxes
