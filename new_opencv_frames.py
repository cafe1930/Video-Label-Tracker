import numpy as np
import cv2

import pandas as pd

from PIL import Image, ImageFont, ImageDraw, ImageColor, ImageFont

from copy import deepcopy

def create_palette(classes_list):
    max_color = 0xFFFFFF
    color_step = max_color // len(classes_list)

    color = 0
    palette = {}
    for person in classes_list:
        # преобразуем целочисленное выражение цвета в RGB кортеж
        r, g, b = ImageColor.getrgb('#{:06x}'.format(color)) # 06x - формат шестиразрядного шестнадцатиричного числа с заполнением нулями пустых разрядов
        palette[person] = (r, g, b)
        color += color_step
    return palette

def check_cursor_in_bbox(x0, y0, x1, y1, cursor_x, cursor_y):
    if cursor_x > x0 and cursor_y > y0 and cursor_x < x1 and cursor_y < y1:
        return True
    return False

def check_cursor_in_corner(corner_x, corner_y, cursor_x, cursor_y, target_radius):
    radius = np.linalg.norm([cursor_x-corner_x, cursor_y-corner_y])
    if radius <= target_radius:
        return True
    return False

def compute_bbox_area(x0,y0,x1,y1):
    return abs(x1-x0)*abs(y1-y0)

def process_box_coords(x0, y0, x1, y1, rows, cols):
    # превращаем строки в числа и фиксируем координаты рамки, чтобы они не выходили за пределы кадра
    x0 = np.clip(int(x0), 0, cols)
    x1 = np.clip(int(x1), 0, cols)
    y0 = np.clip(int(y0), 0, rows)
    y1 = np.clip(int(y1), 0, rows)

    # чтобы у нас ширина и высота рамки была не отрицательной, 
    # переставляем местами нулевую и первую координаты, если первая больше нулевой
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    return x0, y0, x1, y1

def xyxy2xywh(x0, y0, x1, y1):
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    w = max(x0, x1) - min(x0, x1)
    h = max(y0, y1) - min(y0, y1)
    return x0, y0, w, h

def xywh2xyxy(x, y, w, h):
    x0 = x
    y0 = y
    x1 = x + w
    y1 = y + h
    return x0, y0, x1, y1

def compute_iou(coords1, coords2):
    x00, y00, x01, y01 = coords1
    x10, y10, x11, y11 = coords2
    
    x0 = abs(x01 - x00)
    y0 = abs(y01 - y00)

    x1 = abs(x11 - x10)
    y1 = abs(y11 - y10)

    # вычисление площадей отдельных прямоугольников
    s0 = x0*y0
    s1 = x1*y1
    
    # вычисление координат пересечения 
    # если рамки перескаются, то строка или столбец
    # левого верхнего угла одной из рамок должен быть 
    # меньше строки или слобца левого верхнего угла другой рамки соответственно. 
    # Строка или столбец правого нижнего угла одной из рамок должна 
    # быть больше или равна строке или столбцу правого нижнего угла другой рамки соответственно
    x0 = max(x00, x10) 
    y0 = max(y00, y10)
    x1 = min(x01, x11)
    y1 = min(y01, y11)

    # если хотя бы одна из сторон оказалась меньше нуля, значит пересечения нет
    if x1-x0 <= 0 or y1-y0 <= 0:
        s_intersection = 0
    else: 
        s_intersection = (x1-x0)*(y1-y0)

    return s_intersection / (s0 + s1 - s_intersection + 1e-9)
    
def draw_bbox_with_text(
    image:np.array,
    bbox_coords:tuple,
    bbox_width:int,
    class_name:str,
    color:tuple,
    font:ImageFont.FreeTypeFont,
    ):
    '''
    image:np.array - массив пикселей изображения
    bbox_coords:tuple|list, - координаты рамки в формате x0,y0,x1,y1
    bbox_width:int, - ширина рамки
    class_name:str, - имя выводимого класса
    color:tuple|list, - цвет рамки
    font:ImageFont.FreeTypeFont, - шрифт текста
    '''

    x0, y0, x1, y1 = bbox_coords
    image =  Image.fromarray(image)
    cols, rows = image.size
    #x0, y0, x1, y1 = process_box_coords(x0, y0, x1, y1, rows, cols)

    draw = ImageDraw.Draw(image)

    # рисуем прямоугольник для общей рамки...
    draw.rectangle(bbox_coords, outline=color, width=bbox_width)
   

    # определяем цвет шрифта исходя из яркости ЧБ эквивалента цвета класса
    r, g, b = color
    grayscale = int(0.299*r + 0.587*g + 0.114*b)
    # пороговая фильтрация работает на удивление хорошо...
    font_color = 255 if grayscale < 128 else 0

    # вычисляем координаты текста - посередине рамки
    text_coords = ((x1+x0)//2, (y1+y0)//2)

    font_size = font.size

    # квадратный корень почему-то работает очень хорошо для вычисления ширины рамки текста...
    text_bbow_width = np.round(np.sqrt(font_size)).astype(int)
    # вычисляем зазор между рамкой текста и текстом
    text_bbox_spacing = text_bbow_width//3 if text_bbow_width//3 > 1 else 1

    # определяем координаты обрамляющего текст прямоугольника
    text_bbox = draw.textbbox(text_coords, class_name, font=font, anchor='mm') # anchor='mm' означает расположение текста посередине относительно координат
    # расширяем рамку на 3 пикселя в каждом направлении
    text_bbox = tuple(np.add(text_bbox, (-text_bbow_width, -text_bbow_width, text_bbow_width, text_bbow_width)))

    # рисуем прямоугольник для текста
    draw.rectangle(text_bbox, outline=(font_color, font_color, font_color), fill=color, width=bbox_width-text_bbox_spacing)
    # пишем текст
    draw.text(text_coords, class_name, font=font, anchor='mm', fill=(font_color, font_color, font_color))

    return np.array(image)

class Bbox:
    def __init__(
            self,
            x0,
            y0,
            x1,
            y1,
            img_rows,
            img_cols,
            class_name,
            auto_idx,
            registered_idx,
            color,
            displaying_type='auto',
            tracking_type='no'):

        '''
        Класс, описывающий поведение одной локализационной рамки
        x0, y0, x1, y1 - координаты правого верхнего и левого нижнего углов рамки
        img_rows, img_cols - количество строк и стоблцов изображения, необходимые для нормировки координат рамок
        class_name - имя класса
        color - цвет рамки
        sample_idx - индекс или номер рамки одного и того же класса для отслеживания ситуаций, когда на изображении много объектов одного и того же класса 
        displaying_type - отображаемый тип данных: ['auto', 'registered', 'no']
        tracking_type:str - тип трэкинга. Возможные варианты ['yolo', 'opencv', 'no']
        '''

        # координаты левого верхнего и правого нижнего углов рамки
        self.coords = (x0, y0, x1, y1)
        # имя класса
        self.class_name = class_name
        # индекс объекта какого-то определенного класса, полученный из автоматического трекера
        self.auto_idx = auto_idx
        # индекс отслеживаемого объекта
        self.registered_idx = registered_idx
        # цвет рамки
        self.color = color

        # координаты начального угла рамки
        self.ix = None
        self.iy = None

        # координаты смещений углов рамки при создании и изменении
        self.dx0 = None
        self.dy0 = None
        self.dx1 = None
        self.dy1 = None

        # флаг, сигнализирующий, что данная рамка создается
        self.is_bbox_creation = False
        # флаг, сигнализирующий, что координаты какого-либо угла рамки изменяются
        self.is_corner_dragging = False
        # Флаг, сигнализирующий, что рамка перемещается по кадру0
        self.is_bbox_dragging = False

        # размер карда
        self.img_rows = img_rows
        self.img_cols = img_cols

        # флаг для отображения и для трекинга
        self.displaying_type = displaying_type

        self.tracking_type = tracking_type

        
    def __hash__(self):
        '''
        Для применения в словарях и множествах
        '''
        return hash((self.coords, self.class_name, self.id))
    
    def update_color(self, color):
        self.color = color
    
    def get_class_name(self):
        return self.class_name
    
    def get_class_id_str(self):
        return f'{self.class_name},{self.id}'
    
    def x0y0x1y1(self):
        return self.coords
    
    def numpy_coords(self):
        return np.array(self.coords)

    def x0y0wh(self):
        '''
        Перевод координат из формата x0y0x1y1 в формат x0,y0, ширина, высота
        для обеспечения работы трекера opencv
        '''
        x0,y0,x1,y1 = self.coords
        return xyxy2xywh(x0,y0,x1,y1)

    def corner_drag(self, corner_x, corner_y):
        x0, y0, x1, y1 = self.coords
        self.is_bbox_creation = False
        self.is_bbox_dragging = False
        if not self.is_corner_dragging:
            cursor_corner_arr = np.array([corner_x, corner_y])
            bbox_cornenrs_matrix = np.array(
                [[x0,y0],
                 [x0,y1],
                 [x1,y0],
                 [x1,y1]]
            )
            distances_array = np.linalg.norm(bbox_cornenrs_matrix-cursor_corner_arr, axis=1)
            farthest_corner_idx = np.argmax(distances_array)
            self.ix, self.iy = bbox_cornenrs_matrix[farthest_corner_idx]
            
            
            self.is_corner_dragging = True
        self.update_coords(self.ix, self.iy, corner_x, corner_y)

    def create_bbox(self, corner_x, corner_y):
        x0, y0, x1, y1 = self.coords
        self.is_bbox_dragging = False
        self.is_corner_dragging = False

        if not self.is_bbox_creation:
            cursor_corner_arr = np.array([corner_x, corner_y])
            bbox_cornenrs_matrix = np.array(
                [[x0,y0],
                 [x0,y1],
                 [x1,y0],
                 [x1,y1]]
            )
            distances_array = np.linalg.norm(bbox_cornenrs_matrix-cursor_corner_arr, axis=1)
            nearest_corner_idx = np.argmin(distances_array)
            self.ix, self.iy = bbox_cornenrs_matrix[nearest_corner_idx]
            self.is_bbox_creation = True

        self.update_coords(self.ix, self.iy, corner_x, corner_y)

    def compute_initial_corner(self, x0, y0, x1, y1, corner_x, corner_y):
        # определяем, к какому углу ближе курсор
        dist0 = np.linalg.norm([x0-corner_x, y0-corner_y])
        dist1 = np.linalg.norm([x1-corner_x, y1-corner_y])
        if dist0 <= dist1:
            self.ix = x0
            self.iy = y0
        else:
            self.ix = x1
            self.iy = y1

    def box_drag(self, x, y):
        self.is_corner_dragging = False
        self.is_bbox_creation = False
        if not self.is_bbox_dragging:
            x0, y0, x1, y1 = self.coords
            self.dx0 = x - x0
            self.dy0 = y - y0

            self.dx1 = x1 - x
            self.dy1 = y1 - y

            self.is_bbox_dragging = True
        else:
            x0 = x - self.dx0
            y0 = y - self.dy0

            x1 = self.dx1 + x
            y1 = self.dy1 + y
            self.update_coords(x0, y0, x1, y1)

    def stop_corner_drag(self):
        self.is_corner_dragging = False
        self.ix = None
        self.iy = None

    def stop_bbox_creation(self):
        self.is_bbox_creation = False
        self.ix = None
        self.iy = None

    def stop_box_drag(self):
        self.is_bbox_dragging = False
        self.dx0 = None
        self.dy0 = None
        self.dx1 = None
        self.dy1 = None
        
    def update_coords(self, x0, y0, x1, y1):
        self.coords = (x0, y0, x1, y1)
        
    def update_class_name(self, class_name):
        self.class_name = class_name
        
    def make_x0y0_lesser_x1y1(self):
        # превращаем строки в числа и фиксируем координаты рамки, чтобы они не выходили за пределы кадра
        x0, y0, x1, y1 = self.coords

        x0 = np.clip(int(x0), 0, self.img_cols)
        x1 = np.clip(int(x1), 0, self.img_cols)
        y0 = np.clip(int(y0), 0, self.img_rows)
        y1 = np.clip(int(y1), 0, self.img_rows)

        # чтобы у нас ширина и высота рамки была не отрицательной, 
        # переставляем местами нулевую и первую координаты, если первая больше нулевой
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        
        self.coords = (x0, y0, x1, y1)

    def iou_compare(self, other_bbox, iou_threshold):
        iou = compute_iou(self.coords, other_bbox.coords)
        if iou <= iou_threshold:
            return True
        return False
    
    def compute_bbox_area(self):
        return compute_bbox_area(*self.coords)
    
    def __repr__(self) -> str:
        class_name = self.class_name
        #id = self.id
        repr_str = f'{class_name},auto_idx:{self.auto_idx},registered_idx:{self.registered_idx}: {self.coords}, displaying_type={self.displaying_type}, tracking_type={self.tracking_type}'
        return repr_str

class BboxFrameTracker:
    def __init__(self, img, registered_objects_db):
        '''
        Класс служит для создания, изменения и рендеринга множества рамок, локализующих различные объекты в кадре
        Используется в программе Video-Label-Tracker
        Input:
            img: numpy.ndarray, shape=(rows, cols, channels) - кадр видео
        '''
        # на всякий случай копируем кадр
        self.img = img.copy()

        # словарь, где мы храним все рамки
        #self.bboxes_container = {}
        self.bboxes_container = BboxesContainer(registered_objects_db)

        # координаты изменяемого угла
        self.displayed_corner = None
        # координаты изменяемой рамки 
        self.displayed_box = None 

        # рамка, которая в данный момент или создается, или изменяется
        self.processing_box = None

        # имя класса конкретной рамки, с которой мы проводим манипуляции
        self.current_class_name = None

        self.delete_box_flag = False

        # флаг, сигнализирующий о том, что рамка создана. Используется для выбора класса рамки
        self.is_bbox_created = False

        # флаг, сигнализирующий о том, что рамки каким-то образом изменились
        self.is_bboxes_changed = False

        # флаг-костыль, для обработки перемещения рамок
        self.is_bboxes_dragged = False

        # флаг, сигнализирующий о том, что мы показываем номер рамки одного и того же класса
        self.is_bbox_idx_displayed = True
    
    

    def draw_one_box(self, event, flags, x, y):
        '''
        Создание новой рамки вручную
        '''
        rows, cols, channels = self.img.shape
        
        if event == cv2.EVENT_LBUTTONDOWN and not flags & cv2.EVENT_FLAG_CTRLKEY:
            if self.processing_box is None:
                
                color = (0, 0, 0)
                # !!!!!!!
                auto_idx = -1
                registered_idx = -1
                current_class_name = '?'
                self.processing_box = Bbox(
                    x, y, x, y, rows, cols, current_class_name, auto_idx, registered_idx, color)
                self.processing_box.create_bbox(x, y)
                # изменение по корректируемой рамке
                self.bboxes_container.update_bbox(self.processing_box)

        # mouse is being moved, draw rectangle
        elif event == cv2.EVENT_MOUSEMOVE and not flags & cv2.EVENT_FLAG_CTRLKEY:
            if self.processing_box is not None:
                if self.processing_box.is_bbox_creation:
                    # для обеспечения "передвижения" угла рамки, мы постоянно извлекаем станые координаты рамки и добавляем новые
                    self.processing_box.create_bbox(x, y)
                    # изменение по корректируемой рамке
                    self.bboxes_container.update_bbox(self.processing_box)
                    
        # if the left mouse button was released, set the drawing flag to False
        elif event == cv2.EVENT_LBUTTONUP and not flags & cv2.EVENT_FLAG_CTRLKEY:
            # фиксируеми нарисованную рамку
            if self.processing_box is not None and self.processing_box.is_bbox_creation:
                self.processing_box.create_bbox(x, y)
                
                self.processing_box.make_x0y0_lesser_x1y1()
                self.processing_box.stop_bbox_creation()

                # изменение по корректируемой рамке
                self.bboxes_container.update_bbox(self.processing_box)

                self.processing_box = None
                
                # флаг, сигнализирующий о создании новой рамки
                self.is_bbox_created = True
        else:
            #self.is_bboxes_changed = False
            self.is_bbox_created = False

    def correct_rectangle(self, event, flags, bbox, x, y):
        rows, cols, channels = self.img.shape
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            if event == cv2.EVENT_LBUTTONDOWN:
            
                if self.processing_box is None:
                    # здесь self.processing_box должен быть проинициализирован, иначе возвращаемся из функции
                    # извлечение из словаря по заданному имени
                    self.processing_box = bbox
                    self.processing_box.corner_drag(x, y)
                    # изменение по корректируемой рамке
                    # !!!!!
                    self.bboxes_container.update_bbox(self.processing_box)
        
            # mouse is being moved, draw rectangle
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.processing_box is not None:
                    if self.processing_box.is_corner_dragging:
                        # для обеспечения "передвижения" угла рамки, мы постоянно извлекаем станые координаты рамки и добавляем новые
                        self.processing_box.corner_drag(x, y)
                        # изменение по корректируемой рамке
                        self.bboxes_container.update_bbox(self.processing_box)
                
            # if the left mouse button was released, set the drawing flag to False
            elif event == cv2.EVENT_LBUTTONUP:
                if self.processing_box is not None and self.processing_box.is_corner_dragging:
                    # фиксируеми нарисованную рамку
                    self.processing_box.is_corner_dragging = False
                    self.processing_box.corner_drag(x, y)
                    self.processing_box.make_x0y0_lesser_x1y1()
                    self.processing_box.stop_corner_drag()
                    # изменение по корректируемой рамке
                    self.bboxes_container.update_bbox(self.processing_box)
                    self.processing_box = None
                    self.is_bboxes_changed = True
        else:
            if self.processing_box is not None:
                self.processing_box.make_x0y0_lesser_x1y1()
                self.processing_box.stop_corner_drag()
            self.processing_box = None
            self.is_bboxes_changed = False

    def drag_box(self, event, flags, bbox, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                if self.processing_box is None:
                    # извлекаем корректируемую рамку из списка по заданному имени
                    self.processing_box = bbox
                    self.processing_box.box_drag(x, y)
                    # изменение по корректируемой рамке
                    self.bboxes_container.update_bbox(self.processing_box)
        # mouse is being moved, draw rectangle
        elif event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                if self.processing_box is not None:
                    if self.processing_box.is_bbox_dragging:
                        # для обеспечения "передвижения" угла рамки, мы постоянно извлекаем станые координаты рамки и добавляем новые
                        self.processing_box.box_drag(x, y)
                        # изменение по корректируемой рамке
                        self.bboxes_container.update_bbox(self.processing_box)
            else:
                self.processing_box.make_x0y0_lesser_x1y1()
                self.processing_box.stop_box_drag()
                self.processing_box = None                
        # if the left mouse button was released, set the drawing flag to False
        elif event == cv2.EVENT_LBUTTONUP:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                if self.processing_box is not None:
                    if self.processing_box.is_bbox_dragging:
                        self.processing_box.box_drag(x, y)
                        self.processing_box.make_x0y0_lesser_x1y1()
                        self.processing_box.stop_box_drag()

                        # изменение по корректируемой рамке
                        self.bboxes_container.update_bbox(self.processing_box)
                        self.processing_box = None
                        #self.is_bboxes_changed = True
                        self.is_bboxes_dragged = True
            else:
                self.is_bboxes_dragged = False
                
        else:
            #self.is_bboxes_changed = False
            self.is_bboxes_dragged = False
    
    def change_class_name(self, event, flags, bbox):
        if event == cv2.EVENT_RBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY and not flags & cv2.EVENT_FLAG_ALTKEY:
                if self.processing_box is None:
                    if self.current_class_name is not None:
                        # извлечение из словаря по заданному имени!
                        self.processing_box = bbox
                        current_color = (0,255,0)
                        class_name, id = self.current_class_name.split(',')

                        self.processing_box.class_name = class_name
                        self.processing_box.id = int(id)
                        self.processing_box.color = current_color
                        # изменение по информации извне - по заданному имени класса
                        self.bboxes_container.update_bbox(self.processing_box)
                        # удаление рамки по заданному имени
                        #self.bboxes_container.pop(bbox)
                        self.processing_box = None
                        self.is_bboxes_changed = True
        else:
            self.is_bboxes_changed = False

    def delete_box(self, event, flags, bbox):
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_ALTKEY and not flags & cv2.EVENT_FLAG_CTRLKEY:
                if self.processing_box is None:
                    # удаление по заданному имени
                    self.processing_box = self.bboxes_container.pop(bbox)
                    
                    self.processing_box = None
                    self.is_bboxes_changed = True
        else:
            self.is_bboxes_changed = False

    def update_img(self, img):
        self.img = img.copy()

    def delete_img(self):
        self.img = None

    def update_current_class_name(self, current_class_name):
        self.current_class_name = current_class_name

    def __call__(self, event, x, y, flags, param):
        '''
        Обработка коллбэков opencv. Сигнатура метода совпадает с сигнатурой обработчика коллбэков opencv.
        '''
        
        # при зажатом Ctl мы изменяем (перетаскиваем или меняем размер) рамку
        if (flags & cv2.EVENT_FLAG_CTRLKEY)==cv2.EVENT_FLAG_CTRLKEY and not (flags & cv2.EVENT_FLAG_ALTKEY)==cv2.EVENT_FLAG_ALTKEY:
            self.delete_box_flag = False
            if self.processing_box is not None:
                self.displayed_box = self.processing_box.coords
                #bbox_name = self.processing_box.get_class_id_str()
                if self.processing_box.is_corner_dragging:
                    self.correct_rectangle(event, flags, self.displayed_box, x, y)
                elif self.processing_box.is_bbox_dragging:
                    self.drag_box(event, flags, self.displayed_box, x, y)  
            else:
                # итерирование с извлечением имени и рамки
                for bbox in self.bboxes_container.iter_bboxes():
                    x0, y0, x1, y1 = bbox.coords
                    
                    if check_cursor_in_corner(x0,y0,x,y,6):
                        self.displayed_corner = (x0, y0)
                        self.displayed_box = None
                        self.correct_rectangle(event, flags, bbox, x, y)
                        break
                    elif check_cursor_in_corner(x1,y1,x,y,6):
                        self.displayed_corner = (x1, y1)
                        self.displayed_box = None
                        self.correct_rectangle(event, flags, bbox, x, y)
                        break
                    elif check_cursor_in_corner(x0,y1,x,y,6):
                        self.displayed_corner = (x0, y1)
                        self.displayed_box = None    
                        self.correct_rectangle(event, flags, bbox, x, y)
                        break
                    elif check_cursor_in_corner(x1,y0,x,y,6):
                        self.displayed_corner = (x1, y0)
                        self.displayed_box = None
                        self.correct_rectangle(event, flags, bbox, x, y)
                        break

                    elif check_cursor_in_bbox(x0, y0, x1, y1, x, y):
                        self.displayed_box = (x0, y0, x1, y1)
                        self.displayed_corner = None
                        self.drag_box(event, flags, bbox, x, y)
                        self.change_class_name(event, flags, bbox)
                        break
                    else:
                        self.displayed_corner = None
                        self.displayed_box = None
                        #self.is_bboxes_changed = False
        # При зажатом Alt мы удаляем рамку
        elif (flags & cv2.EVENT_FLAG_ALTKEY)==cv2.EVENT_FLAG_ALTKEY and not (flags & cv2.EVENT_FLAG_CTRLKEY)==cv2.EVENT_FLAG_CTRLKEY:
            # итерирование с извлечением имени и рамки
            for bbox in list(self.bboxes_container.iter_bboxes()):
                x0, y0, x1, y1 = bbox.coords
                if check_cursor_in_bbox(x0, y0, x1, y1, x, y):
                    self.displayed_box = (x0, y0, x1, y1)
                    self.displayed_corner = None
                    self.delete_box_flag = True
                    self.delete_box(event, flags, bbox)
                else:
                    self.displayed_box = None
                    self.delete_box_flag = False
                    #self.is_bboxes_changed = False
        else:
            self.delete_box_flag = False
            # фактически, мы вызываем всегда функцию draw_one_box, а уже внутри нее обрабатываем нажатия кнопок
            self.draw_one_box(event, flags, x, y)

    def update_bboxes_container(self, new_bboxes_container):
        # обновление контейнера
        self.bboxes_container = new_bboxes_container

    def render_boxes(self):
        '''
        Метод для отображения рамок на экране
        '''

        #drawing_img = self.img.copy()
        drawing_img = deepcopy(self.img)
        rows, cols, channels = drawing_img.shape
        # определяем размер шрифта исходя из размера изображения
        font_size = min(rows,cols)//30
        # устанавливаем шрифт для указания размечаемых людей
        font = ImageFont.truetype("FiraCode-SemiBold.ttf", font_size)
        # вычисляем ширину рамки. Квадратный корень почему-то работает хорошо...
        line_width = np.round(np.sqrt(font_size).astype(int))
        # итерирование по рамкам в обратном порядке (чтобы те рамки, которые были добавлены последними, рендерились поверх остальных)
        for bbox in list(self.bboxes_container.iter_bboxes())[::-1]:
            if bbox.displaying_type != 'no':
                x0, y0, x1, y1 = bbox.coords
                class_name = bbox.class_name

                if bbox.displaying_type == 'auto':
                    color = (0, 0, 0)
                    displaying_name = f'{class_name}(AG),{bbox.auto_idx}'
                elif bbox.displaying_type == 'registered':
                    color = (0, 255, 0)
                    displaying_name = f'{class_name}(T),{bbox.registered_idx}'
                else:
                    raise ValueError('Bbox.displaying_type shold be either "registered" or "auto" or "no"')
                
                #color = bbox.color
                drawing_img = draw_bbox_with_text(drawing_img, (x0,y0,x1,y1), line_width, displaying_name, color, font)
                # кружок, обозначающий левый верхний угол
                #drawing_img = cv2.circle(drawing_img, (x0, y0), 6, color, -1)
                
                
                if bbox.is_bbox_creation:
                    drawing_img = cv2.circle(drawing_img, (x1, y1), 6, (0, 0, 255), -1)
                elif bbox.is_corner_dragging:
                    if (bbox.ix, bbox.iy) == (x0, y0):
                        drawing_img = cv2.circle(drawing_img, (x1, y1), 6, (0, 0, 255), -1)
                    elif (bbox.ix, bbox.iy) == (x1, y1):
                        drawing_img = cv2.circle(drawing_img, (x0, y0), 6, (0, 0, 255), -1)
                else:
                    # кружок, обозначающий правый нижний угол
                    #drawing_img = cv2.circle(drawing_img, (x1, y1), 6, color, -1)
                    pass

                if self.displayed_corner is not None and not bbox.is_corner_dragging:
                    drawing_img = cv2.circle(drawing_img, self.displayed_corner, 6, (0, 0, 255), -1)

                if self.displayed_box is not None:
                    x0,y0,x1,y1 = self.displayed_box
                    if self.delete_box_flag:
                        thickness = -1
                        drawing_img = cv2.rectangle(drawing_img, (x0, y0), (x1, y1), (0, 0, 255), thickness)
                    else:
                        thickness = 4
                    
                    
        return drawing_img

class BboxesContainer:
    '''
    ВСКРЫЛАСЬ НЕОБХОДИМОСТЬ КАК-ТО ХРАНИТЬ БД СО ВСЕМИ ЗАРЕГИСТРИРОВАННЫМИ ОБЪЕКТАМИ!!!!
    '''
    def __init__(self, registered_objects_db) -> None:
        '''
        registered_objects_db - база данных, где хранится информация об отслеживаемых объектах:
        '''      
        # основная таблица с рамками
        # !!!!! не вполне понятно,зачем теперь нужно поле id, раз я решил осуществлять поиск посредством фильтрации значений полей
        # Колонки у таблицы:
        #   class_name - имя класса
        #   object_description - описание объекта
        #   auto_idx - индекс, присвоенный автоматическим трекером
        #   registered_idx - индекс отслеживаемого объекта
        #   bbox - сам объект рамки
        #   is_updated - флаг, сигнализирующий о том, что рамка обновилась на очередном кадре
        self.bboxes_df = pd.DataFrame(columns=['class_name', 'object_description', 'auto_idx', 'registered_idx', 'bbox', 'is_updated'])


        self.registered_objects_db = registered_objects_db

        # {имя класса: индекс последнего элемента для класса}
        self.tracking_classes_counter = {}

    def find_bbox(self, class_name=None, auto_idx=None, registered_idx=None, object_description=None):
        '''
        Поиск рамок по имени класса, автоматическому индексу, индексу, присвоенному вручную и текстовому описанию объектов
        '''
        filter_condition = ~self.bboxes_df['bbox'].isna()
        
        if class_name is not None:
            filter_condition = filter_condition & (self.bboxes_df['class_name']==class_name)
        if auto_idx is not None:
            filter_condition = filter_condition & (self.bboxes_df['auto_idx']==auto_idx)
        if registered_idx is not None:
            filter_condition = filter_condition & (self.bboxes_df['registered_idx']==registered_idx)
        if object_description is not None:
            filter_condition = filter_condition & (self.bboxes_df['object_description']==object_description)

        return self.bboxes_df[filter_condition]

    def show_bbox_certain_type(self, showing_type):
        '''
        Показывает рамки определенного типа (автоматические или отслеживаемые вручную)
        showing_type = ['auto', 'registered', 'both']
        НЕ ДОПИСАНО!!!!111
        '''
        if showing_type == 'auto':
            # как делать inplace операции с объектами, хранящимися в ячейках?
            self.bboxes_df['bboxes'].apply(lambda x: True if showing_type=='' else x.is_visible)

    def update_bbox(self, updating_bbox):
        '''
        Обновление рамок, включая добавление новых
        updating_bbox - обновляемая рамка
        updting_source - источник обновления из множества ['auto', 'registered']
        '''
        updating_class_name = updating_bbox.class_name
        updating_auto_idx = updating_bbox.auto_idx
        updating_manual_idx = updating_bbox.registered_idx

        # обработка различных источников обновления 
        if updating_auto_idx != -1:
            # автоматически сгенерированная рамка
            # признак рамки - то же самое имя класса и тот же самый автоматический индекс
            # если рамка помечена 'registered', то все равно ищем по признаку автоматического индекса
            # индекс auto затирает индекс registered 
            filter_condition = (self.bboxes_df['class_name']==updating_class_name)\
                & (self.bboxes_df['auto_idx']==updating_auto_idx)
            registered_idx = -1
        elif updating_auto_idx == -1 and updating_manual_idx == -1:
            filter_condition = (self.bboxes_df['class_name']==updating_class_name)\
                & (self.bboxes_df['registered_idx']==updating_manual_idx)\
                & (self.bboxes_df['auto_idx']==updating_auto_idx)
            registered_idx = -1 
        else:
            #if updating_manual_idx == -1:
            #    raise ValueError('updating_manual_idx should not equal to -1 in a case of updating_auto_idx == -1')
            # если у рамки нет автоматического индекса (рамка создана вручную), то тогда 
            # мы ищем рамку по индексу registered
            filter_condition = (self.bboxes_df['class_name']==updating_class_name)\
                & (self.bboxes_df['registered_idx']==updating_manual_idx) 
            registered_idx = updating_bbox.registered_idx
        
        filtered_bboxes_df = self.bboxes_df[filter_condition]
        # если рамки нет БД, ее надо добавить
        if len(filtered_bboxes_df) < 1:
            self.bboxes_df.loc[len(self.bboxes_df)] = {
                'class_name': updating_class_name,
                'object_description': '',
                'auto_idx': updating_auto_idx,
                'registered_idx': registered_idx,
                'bbox': updating_bbox,
                'is_updated':True
                }
        # случай, если рамка есть в БД, и она единственная
        elif len(filtered_bboxes_df) == 1:
            # если автоматическое обновление разрешено
            index = filtered_bboxes_df.index[0]
            
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

    def get_autogenerated_bboxes(self):
        filter_condition = self.bboxes_df['auto_idx'] != -1
        return self.bboxes_df[filter_condition]
        #return [bbox for bbox in self.bboxes_df[filter_condition]['bbox']]

    def get_registered_bboxes(self):
        #filter_condition = self.bboxes_df['registered_idx'] != -1
        return self.registered_objects_db
        #return [bbox for bbox in self.bboxes_df[filter_condition]['bbox']]

    def change_tracking_type(self, bbox, new_tracking_type):
        '''
        Изменить тип отслеживания
        new_tracking_type - тип трекинга объекта ['auto', 'registered']
        '''
        updating_class_name = bbox.class_name
        updating_auto_idx = bbox.auto_idx
        updating_manual_idx = bbox.registered_idx
    
    def check_updated_bboxes(self):
        '''
        Проверка и сохранения только тех рамок, которые были обновлены
        '''
        are_updated_bboxes = self.bboxes_df['is_updated'] == True
        are_registered_bboxes = self.bboxes_df['registered_idx'] != -1
        registered_and_not_updated = ~are_updated_bboxes & are_registered_bboxes
        not_registered_and_not_updated = ~are_updated_bboxes & ~are_registered_bboxes
        for idx, row in self.bboxes_df[registered_and_not_updated].iterrows():
            bbox = row['bbox']
            bbox.displaying_type = 'no'
            row['bbox'] = bbox
        
        dissapeared_bboxes = self.bboxes_df[not_registered_and_not_updated]
        self.bboxes_df = self.bboxes_df[~not_registered_and_not_updated]
        # выставляем все рамки как не обновляемые
        self.bboxes_df = self.bboxes_df.assign(is_updated=False)
        return dissapeared_bboxes
    
    def change_bboxes_displaying_type(self, displaying_type):
        '''
        Изменение типа отображения рамок
        displaying_type - тип отображения ['auto', 'registered', 'full']
            'auto' - только автоматически сгенерированные
            'registered' - только зарегистрированные
            'full' - полные
        '''       
        for i, row in self.bboxes_df.iterrows():
            bbox = row['bbox']
            if displaying_type == 'full':
                if row['registered_idx'] != -1:
                    bbox.displaying_type = 'registered'
                    bbox.color = (0, 255, 0)
                else:
                    bbox.displaying_type = 'auto'
                    bbox.color = (0, 0, 0)
            elif displaying_type == 'auto':
                bbox.displaying_type = 'auto'
                bbox.color = (0, 0, 0)
            elif displaying_type == 'registered':
                if row['registered_idx'] != -1:
                    bbox.displaying_type = 'registered'
                    bbox.color = (0, 255, 0)
                else:
                    bbox.displaying_type = 'no'
                    bbox.color = (0, 0, 0)
            else:
                raise ValueError('displaying_type should be either "auto" either "registered" or "full"')

            row['bbox'] = bbox
            
    def pop(self, bbox):
        '''
        Извлечение рамки из контейнера 
        '''
        class_name = bbox.class_name
        auto_idx = bbox.auto_idx
        registered_idx = bbox.registered_idx

        filter_condition = (self.bboxes_df['class_name']==class_name)\
                & (self.bboxes_df['auto_idx']==auto_idx)\
                & (self.bboxes_df['registered_idx']==registered_idx)
        
        filtered_bboxes_df = self.bboxes_df[filter_condition]

        index = filtered_bboxes_df.index[0]
        self.bboxes_df = self.bboxes_df.drop(index=index)
        
        return filtered_bboxes_df

    def get_autobbox(self, class_name, auto_idx):
        filter_condition = (self.bboxes_df['class_name']==class_name)\
                & (self.bboxes_df['auto_idx']==auto_idx)
        
        return self.bboxes_df[filter_condition].iloc[0]
        
    def get_manualbbox(self, class_name, registered_idx):
        filter_condition = (self.bboxes_df['class_name']==class_name)\
                & (self.bboxes_df['auto_idx']==registered_idx)
        
        return self.bboxes_df[filter_condition].iloc[0]

    def get_auto_bbox_from_registered(self, class_name, registered_idx, object_descr):
        filter_condition = (self.bboxes_df['class_name']==class_name)\
                & (self.bboxes_df['registered_idx']==registered_idx)\
                & (self.bboxes_df['object_description']==object_descr)
        print('DEBUG get_auto_bbox_from_registered')
        print(f'class_name={class_name}; registered_idx={registered_idx}; object_descr={object_descr}')
        #print(self.bboxes_df)
        #print(filter_condition)
        return self.bboxes_df[filter_condition]

    def append_to_tracking_objects_db(self, class_name, object_description, path_to_db):
        '''
        Добавляем новый объект в базу данных отслеживаемых объектов
        '''
        #print('DEBUG')
        #print(f'object_description={object_description}')
        #for obj_descr in self.registered_objects_db['object_description']:
        #    print(f'{obj_descr}:{obj_descr==object_description}')
        #print(f"{object_description in self.registered_objects_db['object_description']}")

        if object_description != '' and object_description in self.registered_objects_db['object_description'].values:
            return -1
        
        self.registered_objects_db = self.registered_objects_db.sort_values(by=['class_name', 'object_idx']).reset_index(drop=True)
        # выясняем индекс класса добавляемого объекта
        
        idx_of_appending_obj = len(self.registered_objects_db[self.registered_objects_db['class_name']==class_name])
        appending_row = {
            'object_idx':idx_of_appending_obj,
            'class_name':class_name,
            'object_description':object_description
            }
        # добавляем строку в таблицу
        self.registered_objects_db.loc[len(self.registered_objects_db)] = appending_row

        # сохраняем результат на диск
        self.registered_objects_db.to_csv(path_to_db, index=False)

        return 0

    def delete_from_tracking_objects_db(self, class_name, object_idx, object_description, path_to_db):
        filter_condition = (self.registered_objects_db['object_idx']==object_idx)\
                & (self.registered_objects_db['class_name']==class_name)\
                & (self.registered_objects_db['object_description']==object_description)
        

        index = self.registered_objects_db[filter_condition].index        
        self.registered_objects_db = self.registered_objects_db.drop(index=index)

        # сохраняем результат на диск
        self.registered_objects_db.to_csv(path_to_db, index=False)

    def find_in_tracking_objects_db(self, class_name, object_description):
        '''
        Поиск в БД отслеживемых объектов по имени класса и описанию объектов
        '''
        filter_condition = (self.registered_objects_db['class_name']==class_name)\
                & (self.registered_objects_db['object_description']==object_description)
        return self.registered_objects_db[filter_condition]

    def get_tracking_obj_idx(self, class_name, object_description):
        filter_condition = (self.registered_objects_db['class_name']==class_name)\
                & (self.registered_objects_db['object_description']==object_description)
        return self.registered_objects_db[filter_condition]['object_idx']

    def assocoate_auto_bbox_with_registered_object(self, class_name, auto_idx, object_description, registered_idx):
        '''
        Выполняется изменение registered_idx и добавление описания объекта
        '''

        # ищем автоматически сгенерированную рамку
        auto_bbox_filter_condition = (self.bboxes_df['class_name']==class_name)\
                & (self.bboxes_df['auto_idx']==auto_idx)
        filtered_df = self.bboxes_df[auto_bbox_filter_condition]
        if len(filtered_df) > 1:
            raise ValueError('Only bboxes with unique pair <class_name, autogenerated_idx> ought to be in self.bboxes_df')
        elif len(filtered_df) == 0:
            raise ValueError('Bbox did not found')
        
        # ищем, есть ли уже для того же самого зарегистрированного объекта автоматическая рамка
        previous_registered_bbox_filter_condition = (self.bboxes_df['class_name']==class_name)\
            & (self.bboxes_df['registered_idx']==registered_idx)\
            & (self.bboxes_df['object_description']==object_description)
        previous_registered_df = self.bboxes_df[previous_registered_bbox_filter_condition]
        if len(previous_registered_df) != 0:
            #print('ЗАМЕНА ИМЕЮЩЕЙСЯ РАМКИ ДО УДАЛЕНИЯ')
            #print(self.bboxes_df)
            index = previous_registered_df.index
            # почему-то иначе объъект не изменяется
            bbox = previous_registered_df['bbox'].iloc[0]
            bbox.registered_idx = -1
            bbox.displaying_type = 'auto'
            bbox.color = (0, 0, 0)
            self.bboxes_df.loc[index, 'bbox'] = bbox
            
            self.bboxes_df.loc[index, 'registered_idx'] = -1
            self.bboxes_df.loc[index, 'object_description'] = ''
            #print('ЗАМЕНА ИМЕЮЩЕЙСЯ РАМКИ ПОСЛЕ УДАЛЕНИЯ')
            #print(self.bboxes_df)
        
        # 

        # обновляем на актуальную информацию
        index = filtered_df.index
        # почему-то иначе объъект не изменяется
        bbox = filtered_df['bbox'].iloc[0]
        bbox.registered_idx = registered_idx
        bbox.displaying_type = 'registered'
        bbox.color = (0, 255, 0)
        self.bboxes_df.loc[index, 'bbox'] = bbox
        
        self.bboxes_df.loc[index, 'registered_idx'] = registered_idx
        self.bboxes_df.loc[index, 'object_description'] = object_description
        #print('ЗАМЕНА ИМЕЮЩЕЙСЯ РАМКИ ПОСЛЕ ВСТАВКИ')
        #print(self.bboxes_df)
       
    def iter_bboxes(self):
        '''
        реализация итерирования по рамкам
        '''
        for idx, row in self.bboxes_df.iterrows():
            yield row['bbox']

if __name__ == '__main__':
    # Первая итерация новые рамки, полученные от "нейронки"
    bbox1 = Bbox(1, 2, 3, 4, 100, 100, class_name='person', auto_idx=1, registered_idx=-1, color=(0,0,0), tracking_type='yolo')
    bbox2 = Bbox(1, 4, 8, 8, 100, 100, class_name='person', auto_idx=2, registered_idx=-1, color=(0,0,0), tracking_type='yolo')
    bbox3 = Bbox(6, 6, 6, 6, 100, 100, class_name='person', auto_idx=3, registered_idx=-1, color=(0,0,0), tracking_type='yolo')
    
    
    # новая рамка, которая создана полностью вручную
    #bbox5 = Bbox(7, 8, 9, 10, 100, 100, class_name='person', auto_idx=-1, registered_idx=1, color=(0,255,0), tracking_type='opencv')
    # add
    registered_objects_db = pd.DataFrame(columns=['object_idx', 'class_name', 'object_descr'])
    b_c = BboxesContainer(registered_objects_db)
    
    b_c.update_bbox(bbox1)
    b_c.update_bbox(bbox2)
    b_c.update_bbox(bbox3)

    dissapeared_bboxes = b_c.check_updated_bboxes()
    print(b_c.bboxes_df)
    print()
    # register
    b_c.assocoate_auto_bbox_with_registered_object(class_name='person', auto_idx=1, object_description='ПИДОРАС')
    b_c.assocoate_auto_bbox_with_registered_object(class_name='person', auto_idx=2, object_description='ПИДОРАСИНА')
    
    print(b_c.bboxes_df)
    print()
    # новая рамка из автоматического трекера
    bbox4 = Bbox(4, 5, 6, 7, 100, 100, class_name='person', auto_idx=1, registered_idx=-1, color=(0,255,0), tracking_type='opencv')
    # новая рамка, которая создана полностью вручную
    bbox5 = Bbox(7, 8, 9, 10, 100, 100, class_name='person', auto_idx=-1, registered_idx=1, color=(0,255,0), tracking_type='opencv')
    # update 2nd time
    b_c.update_bbox(bbox4)
    b_c.update_bbox(bbox2)
    b_c.update_bbox(bbox3)

    dissapeared_bboxes = b_c.check_updated_bboxes()
    print(b_c.find_bbox(class_name='person', auto_idx=1, registered_idx=1, object_description='ПИДОРАС'))
    print()