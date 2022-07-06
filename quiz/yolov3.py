import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf


class YOLO_V3:
    def __init__(self):
        f = open('../models/yolov3/classes.txt')
        self.class_names = [str(i).strip() for i in f.readlines()]
        f.close()

    def convolutional(self, input_layer, filters_shape, downsample=False, activate=True, bn=True):
        if downsample:
            input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
            padding = 'valid'
            strides = 2
        else:
            strides = 1
            padding = 'same'

        conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strides,
                      padding=padding, use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                      bias_initializer=tf.constant_initializer(0.))(input_layer)
        if bn:
            conv = tf.keras.layers.BatchNormalization()(conv)
        if activate:
            conv = tf.keras.layers.LeakyReLU(alpha=0.1)(conv)
        return conv

    def residual_block(self, input_layer, input_channel, filter_num1, filter_num2):
        short_cut = input_layer
        conv = self.convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1))
        conv = self.convolutional(conv, filters_shape=(3, 3, filter_num1, filter_num2))

        residual_output = short_cut + conv
        return residual_output

    def darknet53(self, input_data):
        input_data = self.convolutional(input_data, (3, 3, 3, 32))
        input_data = self.convolutional(input_data, (3, 3, 32, 64), downsample=True)

        for i in range(1):
            input_data = self.residual_block(input_data, 64, 32, 64)

        input_data = self.convolutional(input_data, (3, 3, 64, 128), downsample=True)

        for i in range(2):
            input_data = self.residual_block(input_data, 128, 64, 128)

        input_data = self.convolutional(input_data, (3, 3, 128, 256), downsample=True)

        for i in range(8):
            input_data = self.residual_block(input_data, 256, 128, 256)

        route_1 = input_data
        input_data = self.convolutional(input_data, (3, 3, 256, 512), downsample=True)

        for i in range(8):
            input_data = self.residual_block(input_data, 512, 256, 512)

        route_2 = input_data
        input_data = self.convolutional(input_data, (3, 3, 512, 1024), downsample=True)

        for i in range(4):
            input_data = self.residual_block(input_data, 1024, 512, 1024)

        return route_1, route_2, input_data

    def upsample(self, input_layer):
        return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')

    def decode(self, conv_output, i=0):
        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 85))

        conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, 80), axis=-1)

        xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
        xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)
        xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * [8, 16, 32][i]
        pred_wh = (tf.exp(conv_raw_dwdh) * [[[1.25, 1.625], [2, 3.75], [4.125, 2.875]], [[1.875, 3.8125], [3.875, 2.8125], [3.6875, 7.4375]], [[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]]][i]) * [8, 16, 32][i]

        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def build(self):
        input_layer = tf.keras.layers.Input([416, 416, 3])

        route_1, route_2, conv = self.darknet53(input_layer)

        conv = self.convolutional(conv, (1, 1, 1024, 512))
        conv = self.convolutional(conv, (3, 3, 512, 1024))
        conv = self.convolutional(conv, (1, 1, 1024, 512))
        conv = self.convolutional(conv, (3, 3, 512, 1024))
        conv = self.convolutional(conv, (1, 1, 1024, 512))
        conv_lobj_branch = self.convolutional(conv, (3, 3, 512, 1024))

        conv_lbbox = self.convolutional(conv_lobj_branch, (1, 1, 1024, 255), activate=False, bn=False)

        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv = self.upsample(conv)

        conv = tf.concat([conv, route_2], axis=-1)
        conv = self.convolutional(conv, (1, 1, 768, 256))
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv_mobj_branch = self.convolutional(conv, (3, 3, 256, 512))

        conv_mbbox = self.convolutional(conv_mobj_branch, (1, 1, 512, 255), activate=False, bn=False)

        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv = self.upsample(conv)

        conv = tf.concat([conv, route_1], axis=-1)
        conv = self.convolutional(conv, (1, 1, 384, 128))
        conv = self.convolutional(conv, (3, 3, 128, 256))
        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv = self.convolutional(conv, (3, 3, 128, 256))
        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv_sobj_branch = self.convolutional(conv, (3, 3, 128, 256))

        conv_sbbox = self.convolutional(conv_sobj_branch, (1, 1, 256, 255), activate=False, bn=False)

        conv_tensors = [conv_sbbox, conv_mbbox, conv_lbbox]

        output_tensors = []
        for i, conv_tensor in enumerate(conv_tensors):
            pred_tensor = self.decode(conv_tensor, i)
            output_tensors.append(pred_tensor)

        self.model = tf.keras.Model(input_layer, output_tensors)

    def load(self):
        tf.keras.backend.clear_session()
        range1 = 75
        range2 = [58, 66, 74]

        with open('../models/yolov3/yolov3.weights', 'rb') as wf:
            major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

            j = 0
            for i in range(range1):
                if i > 0:
                    conv_layer_name = 'conv2d_%d' % i
                else:
                    conv_layer_name = 'conv2d'

                if j > 0:
                    bn_layer_name = 'batch_normalization_%d' % j
                else:
                    bn_layer_name = 'batch_normalization'

                conv_layer = self.model.get_layer(conv_layer_name)
                filters = conv_layer.filters
                k_size = conv_layer.kernel_size[0]
                in_dim = conv_layer.input_shape[-1]

                if i not in range2:
                    bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                    bn_layer = self.model.get_layer(bn_layer_name)
                    j += 1
                else:
                    conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

                conv_shape = (filters, in_dim, k_size, k_size)
                conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
                conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

                if i not in range2:
                    conv_layer.set_weights([conv_weights])
                    bn_layer.set_weights(bn_weights)
                else:
                    conv_layer.set_weights([conv_weights, conv_bias])

    def image_preprocess(self, image):
        h, w, _ = image.shape

        scale = min(416 / w, 416 / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        image_paded = np.full(shape=[416, 416, 3], fill_value=128.0)
        dw, dh = (416 - nw) // 2, (416 - nh) // 2
        image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
        image_paded = image_paded / 255.

        return image_paded

    def draw_bbox(self, image, bboxes):
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / 80, 1., 1.) for x in range(len(self.class_names))]

        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for i, bbox in enumerate(bboxes):
            coor = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            class_ind = int(bbox[5])
            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 1000)
            if bbox_thick < 1: bbox_thick = 1
            fontScale = 0.75 * bbox_thick
            (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick * 2)

            score_str = " {:.2f}".format(score)

            label = "{}".format(self.class_names[class_ind]) + score_str
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, thickness=bbox_thick)
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)
            cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (255, 255, 0), bbox_thick, lineType=cv2.LINE_AA)
        return image

    def postprocess_boxes(self, pred_bbox, original_image, input_size, score_threshold):
        valid_scale = [0, np.inf]
        pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5, pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)

        org_h, org_w = original_image.shape[:2]
        resize_ratio = min(input_size / org_w, input_size / org_h)

        dw = (input_size - resize_ratio * org_w) / 2
        dh = (input_size - resize_ratio * org_h) / 2

        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]), np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > score_threshold
        mask = np.logical_and(scale_mask, score_mask)
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

        return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

    def bboxes_iou(self, boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

        return ious

    def nms(self, bboxes, iou_threshold):
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]
            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])

                iou = self.bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes

    def predict(self, image):
        original_image = image.copy()
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = self.image_preprocess(np.copy(original_image))
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        pred_bbox = self.model.predict(image_data)

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = self.postprocess_boxes(pred_bbox, original_image, 416, 0.3)
        bboxes = self.nms(bboxes, 0.45)

        image = self.draw_bbox(original_image, bboxes)

        return image


if __name__ == '__main__':
    model = YOLO_V3()
    model.build()
    model.load()
    path = '../data/images/b001a7ce-5cbc6e0b.jpg'
    img = cv2.imread(path)
    result = model.predict(img)
    cv2.imshow('image', img)
    cv2.imshow('result', result)
    cv2.waitKey(0)