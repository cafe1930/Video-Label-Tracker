from new_opencv_frames import Bbox
import pandas as pd


class bboxes_container:
    def __init__(self) -> None:
        self.idx = 0
        # структура: {id: bbox}

        self.id2bbox = {}

        self.bboxes_df = pd.DataFrame(columns=['id', 'class_name', 'auto_idx', 'manual_idx', 'bbox'])

        self.id2auto_names = {}
        self.id2manual_names = {}

        # Какова его структура?
        self.classes_counter = {}
        self.auto_bboxes_indices = {}
        self.tracking_bboxes_indices = {}        

    def update(self, bbox):
        class_name = bbox.class_name
        auto_idx = bbox.autogen_idx
        manual_idx = bbox.manual_idx

        filter_condition = (self.bboxes_df['class_name']==class_name)\
            & (self.bboxes_df['auto_idx']==auto_idx)\
            & (self.bboxes_df['manual_idx']==manual_idx)
        
        filtered_bboxes = self.bboxes_df[filter_condition]
        return filtered_bboxes

    def pop(self, bbox):
        if bbox.manual_idx is not None:
            pass

    def __iter__(self):
        pass



class A:
    l = [1, 2, 3]
    def __iter__(self):
        for v in self.l:
            yield v



if __name__ == '__main__':
    bbox1 = Bbox(1, 2, 3, 4, 100, 100, class_name='person', autogen_idx=1, manual_idx=-1, color=(0,0,0), tracking_type='yolo')
    bbox2 = Bbox(4, 5, 6, 7, 100, 100, class_name='person', autogen_idx=1, manual_idx=1, color=(0,0,0), tracking_type='yolo')
    bbox3 = Bbox(7, 8, 9, 10, 100, 100, class_name='person', autogen_idx=-1, manual_idx=1, color=(0,0,0), tracking_type='yolo')
    b_c = bboxes_container()
    df = pd.DataFrame(columns=['class_name', 'auto_idx', 'manual_idx', 'bbox'])
    new_df = pd.DataFrame([
        {'class_name':bbox1.class_name, 'auto_idx': bbox1.autogen_idx, 'manual_idx':bbox1.manual_idx, 'bbox':bbox1},
        {'class_name':bbox2.class_name, 'auto_idx': bbox2.autogen_idx, 'manual_idx':bbox2.manual_idx, 'bbox':bbox2},
        {'class_name':bbox3.class_name, 'auto_idx': bbox3.autogen_idx, 'manual_idx':bbox3.manual_idx, 'bbox':bbox3}])
    
    df = pd.concat([df, new_df])
    class_name = 'person'
    auto_idx = 1
    manual_idx = 1
    filtering_condition = (df['class_name']==class_name) & (df['auto_idx']==auto_idx) & (df['manual_idx']==manual_idx)
    print(df[filtering_condition]['bbox'].iloc[0])
    print(df)
