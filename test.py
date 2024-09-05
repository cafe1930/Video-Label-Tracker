from new_opencv_frames import Bbox
import pandas as pd


class bboxes_container:
    def __init__(self) -> None:
        self.idx = 0
        # структура: {id: bbox}

        self.id2bbox = {}

        # основная таблица с рамками
        # !!!!! не вполне понятно,зачем теперь нужно поле id, раз я решил осуществлять поиск посредством фильтрации значений полей
        self.bboxes_df = pd.DataFrame(columns=['class_name', 'auto_idx', 'manual_idx', 'bbox', 'is_updated'])
        # вспомогательное множество с теми рамками, которые мы отслеживаем.
        # нужно для обновления совокупности рамок при переходе между кадрами
        self.updated_bboxes_set = set()

        # Какова его структура?
        self.classes_counter = {}
        self.auto_bboxes_indices = {}
        self.tracking_bboxes_indices = {}        

    def update_bbox(self, updating_bbox, updating_source):
        '''
        updating_bbox - обновляемая рамка
        updting_source - источник обновления из множества ['auto', 'manual']
        '''
        updating_class_name = updating_bbox.class_name
        updating_idx = updating_bbox.auto_idx
        
        # обработка различных источников обновления 
        if updating_source == 'auto':
            filter_condition = (self.bboxes_df['class_name']==updating_class_name)\
                & (self.bboxes_df['auto_idx']==updating_idx)
        elif updating_source == 'manual':
            filter_condition = (self.bboxes_df['class_name']==updating_class_name)\
                & (self.bboxes_df['manual_idx']==updating_idx)
        
        filtered_bboxes_df = self.bboxes_df[filter_condition]
        # если рамки нет БД, ее надо добавить
        if len(filtered_bboxes_df) < 1:
            self.bboxes_df.loc[len(self.bboxes_df)] = {
                'class_name': updating_class_name,
                'auto_idx': updating_idx,
                'manual_idx': updating_bbox.manual_idx,
                'bbox': updating_bbox,
                'is_updated':True
                }
        # случай, если рамка есть в БД, и она единственная
        elif len(filtered_bboxes_df) == 1:
            # если автоматическое обновление разрешено
            found_bbox = self.bboxes_df[filter_condition]['bbox'].iloc[0]
            if found_bbox.tracking_type != 'no':
                found_bbox.coords = updating_bbox.coords
                # !!!
                self.bboxes_df[filter_condition]['is_updated'].iloc[0] = True
                # !!!!!
                self.bboxes_df[filter_condition]['bbox'].iloc[0] = found_bbox
        else:
            error_str = f'More than one unique bboxes found in the table bboxes_df\n{filtered_bboxes_df}'
            raise ValueError(error_str)

    def check_updated_bboxes(self):
        '''
        Проверка и сохранения только тех рамок, которые были обновлены
        '''
        filter_condition = self.bboxes_df['is_updated'] == True
        self.bboxes_df = self.bboxes_df[filter_condition]
        # выставляем все рамки как не обновляемые
        self.bboxes_df = self.bboxes_df.assign(is_updated=False)

    def pop(self, bbox):
        if bbox.manual_idx is not None:
            pass

    def __iter__(self):
        pass


if __name__ == '__main__':
    bbox1 = Bbox(1, 2, 3, 4, 100, 100, class_name='person', auto_idx=1, manual_idx=-1, color=(0,0,0), tracking_type='yolo')
    bbox2 = Bbox(1, 4, 8, 8, 100, 100, class_name='person', auto_idx=2, manual_idx=-1, color=(0,0,0), tracking_type='yolo')
    bbox3 = Bbox(6, 6, 6, 6, 100, 100, class_name='person', auto_idx=3, manual_idx=-1, color=(0,0,0), tracking_type='yolo')
    
    bbox4 = Bbox(4, 5, 6, 7, 100, 100, class_name='person', auto_idx=1, manual_idx=1, color=(0,0,0), tracking_type='opencv')
    bbox5 = Bbox(7, 8, 9, 10, 100, 100, class_name='person', auto_idx=-1, manual_idx=1, color=(0,0,0), tracking_type='opencv')

    b_c = bboxes_container()
        
    b_c.update_bbox(bbox1, 'auto')
    b_c.update_bbox(bbox2, 'auto')
    b_c.update_bbox(bbox3, 'auto')
    b_c.check_updated_bboxes()
    print(b_c.bboxes_df)

    b_c.update_bbox(bbox2, 'auto')
    b_c.update_bbox(bbox1, 'auto')
    print()
    print(b_c.bboxes_df)
    

    
    
