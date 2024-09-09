from new_opencv_frames import Bbox
import pandas as pd


class bboxes_container:
    def __init__(self) -> None:
        self.idx = 0
        # структура: {id: bbox}

        self.id2bbox = {}

        # основная таблица с рамками
        # !!!!! не вполне понятно,зачем теперь нужно поле id, раз я решил осуществлять поиск посредством фильтрации значений полей
        # Колонки у таблицы:
        #   class_name - имя класса
        #   object_description - описание объекта
        #   auto_idx - индекс, присвоенный автоматическим трекером
        #   manual_idx - индекс отслеживаемого объекта
        #   bbox - сам объект рамки
        #   is_updated - флаг, сигнализирующий о том, что рамка обновилась на очередном кадре
        self.bboxes_df = pd.DataFrame(columns=['class_name', 'object_description', 'auto_idx', 'manual_idx', 'bbox', 'is_updated'])
        # вспомогательное множество с теми рамками, которые мы отслеживаем.
        # нужно для обновления совокупности рамок при переходе между кадрами
        self.updated_bboxes_set = set()

        # Какова его структура?
        self.auto_bboxes_indices = {}
        # {имя класса: индекс последнего}
        self.tracking_classes_counter = {}

    def show_bbox_certain_type(self, showing_type):
        '''
        Показывает рамки определенного типа (автоматические или отслеживаемые вручную)
        showing_type = ['auto', 'manual', 'both']
        НЕ ДОПИСАНО!!!!111
        '''
        if showing_type == 'auto':
            # как делать inplace операции с объектами, хранящимися в ячейках?
            self.bboxes_df['bboxes'].apply(lambda x: True if showing_type=='' else x.is_visible)

    def update_bbox(self, updating_bbox, updating_source):
        '''
        Обновление рамок, включая добавление новых
        updating_bbox - обновляемая рамка
        updting_source - источник обновления из множества ['auto', 'manual']
        '''
        updating_class_name = updating_bbox.class_name
        updating_auto_idx = updating_bbox.auto_idx
        
        # обработка различных источников обновления 
        if updating_source == 'auto':
            filter_condition = (self.bboxes_df['class_name']==updating_class_name)\
                & (self.bboxes_df['auto_idx']==updating_auto_idx)
            manual_idx = -1
        elif updating_source == 'manual':
            filter_condition = (self.bboxes_df['class_name']==updating_class_name)\
                & (self.bboxes_df['manual_idx']==updating_auto_idx)
            manual_idx = updating_bbox.manual_idx
        
        filtered_bboxes_df = self.bboxes_df[filter_condition]
        # если рамки нет БД, ее надо добавить
        if len(filtered_bboxes_df) < 1:
            self.bboxes_df.loc[len(self.bboxes_df)] = {
                'class_name': updating_class_name,
                'object_description': '',
                'auto_idx': updating_auto_idx,
                'manual_idx': manual_idx,
                'bbox': updating_bbox,
                'is_updated':True
                }
        # случай, если рамка есть в БД, и она единственная
        elif len(filtered_bboxes_df) == 1:
            # если автоматическое обновление разрешено
            index = filtered_bboxes_df.index[0]
            
            #print(filtered_bboxes_df.index)
            found_bbox = filtered_bboxes_df.loc[index]['bbox']
            #filtered = self.bboxes_df[filter_condition]['bbox'].iloc[0]
            if found_bbox.tracking_type != 'no':
                found_bbox.coords = updating_bbox.coords
                
                # !!!
                #self.bboxes_df[filter_condition][index, 'is_updated'] = True
                self.bboxes_df.loc[index, 'is_updated'] = True
                # !!!!!
                #self.bboxes_df[filter_condition][index, 'bbox'] = found_bbox
                self.bboxes_df.loc[index, 'bbox'] = found_bbox
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
        '''
        Извлечение рамки из контейнера 
        '''
        class_name = bbox.class_name
        auto_idx = bbox.auto_idx
        manual_idx = bbox.manual_idx

        filter_condition = (self.bboxes_df['class_name']==class_name)\
                & (self.bboxes_df['auto_idx']==auto_idx)\
                & (self.bboxes_df['manual_idx']==manual_idx)
        
        filtered_bboxes_df = self.bboxes_df[filter_condition]

    def get_autobbox(self, class_name, auto_idx):
        filter_condition = filter_condition = (self.bboxes_df['class_name']==class_name)\
                & (self.bboxes_df['auto_idx']==auto_idx)
        
        return self.bboxes_df[filter_condition].iloc[0]
        
    def get_manualbbox(self, class_name, manual_idx):
        filter_condition = filter_condition = (self.bboxes_df['class_name']==class_name)\
                & (self.bboxes_df['auto_idx']==manual_idx)
        
        return self.bboxes_df[filter_condition].iloc[0]

       
    def iter_bboxes(self):
        '''
        реализация итерирования по рамкам
        '''
        for idx, row in self.bboxes_df.iterrows():
            yield row['bbox']

class ClassObjectsIndexer:
    def __init__(self):
        self.conter_dict = {}
    
    def append_object(self, class_name):
        try:
            self.conter_dict[class_name] += 1
        except:
            self.conter_dict[class_name] = 1

    def pop_object(self, class_name):
        try:
            self.conter_dict[class_name] -= 1
            if self.conter_dict[class_name] <= 0:
                self.conter_dict.pop(class_name)
        except:
            pass

    def __repr__(self):
        return f'Number of class objects:\n{self.conter_dict}'




if __name__ == '__main__':
    # Первая итерация новые рамки, полученные от "нейронки"
    bbox1 = Bbox(1, 2, 3, 4, 100, 100, class_name='person', auto_idx=1, manual_idx=-1, color=(0,0,0), tracking_type='yolo')
    bbox2 = Bbox(1, 4, 8, 8, 100, 100, class_name='person', auto_idx=2, manual_idx=-1, color=(0,0,0), tracking_type='yolo')
    bbox3 = Bbox(6, 6, 6, 6, 100, 100, class_name='person', auto_idx=3, manual_idx=-1, color=(0,0,0), tracking_type='yolo')
    
    '''
    # зарегистрированная рамка
    bbox4 = Bbox(4, 5, 6, 7, 100, 100, class_name='person', auto_idx=1, manual_idx=1, color=(0,255,0), tracking_type='opencv')
    # новая рамка, которая создана полностью вручную
    bbox5 = Bbox(7, 8, 9, 10, 100, 100, class_name='person', auto_idx=-1, manual_idx=1, color=(0,255,0), tracking_type='opencv')
    '''

    b_c = bboxes_container()
    
    b_c.update_bbox(bbox1, 'auto')
    b_c.update_bbox(bbox2, 'auto')
    b_c.update_bbox(bbox3, 'auto')

    b_c.check_updated_bboxes()
    print(b_c.bboxes_df['class_name'].value_counts())
    #print()

    df = b_c.bboxes_df
    condition = (df['class_name']=='person') & (df['auto_idx']==3) & (df['manual_idx']==-1)

    new_df = df[condition]

    b_c.update_bbox(bbox2, 'auto')
    b_c.update_bbox(bbox1, 'auto')
    b_c.check_updated_bboxes()
    #print()
    #print(b_c.bboxes_df)

    '''
    class_indexer = ClassObjectsIndexer()
    class_indexer.append_object('person')
    class_indexer.append_object('person')
    class_indexer.append_object('ПИДОРАС')
    class_indexer.append_object('ПИДОРАСИНА')
    print(class_indexer)
    class_indexer.pop_object('person')
    class_indexer.pop_object('person')
    class_indexer.pop_object('person')
    print(class_indexer)
    '''