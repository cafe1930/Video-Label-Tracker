from new_opencv_frames import Bbox

class bboxes_container:
    def __init__(self) -> None:
        self.idx = 0
        self.container = {}
        self.classes_counter = {}

    def add(self, bbox):
        self.container[self.idx] = bbox
        self.idx += 1


if __name__ == '__main__':
    bbox = Bbox(1, 2, 3, 4, 100, 100, class_name='person', autogen_idx=1, manual_idx=None, color=(0,0,0), tracking_type='yolo')
    b_c = bboxes_container()
    b_c.add(bbox)
