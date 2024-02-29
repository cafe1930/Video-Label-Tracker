import sys
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QMutex, QObject
from PyQt5.QtWidgets import (QWidget, QComboBox, QPushButton, QTableWidgetItem, QLabel, QCheckBox, QAction,
                             QTableWidget, QLCDNumber, QSlider, QListWidget, QAbstractItemView, QMessageBox,
                             QHBoxLayout, QFileDialog, QVBoxLayout, QApplication, QMainWindow, QGridLayout, QListWidgetItem)

import cv2
import os
import glob
import json


import time

from opencv_frames import BboxFrame, Bbox, BboxFrameTracker, compute_iou

from video_label_tracker import ImshowThread, SetFrameDialog

class LabelViewerWindow(QMainWindow):
    def __init__(self, screen_width, screen_height):
        super().__init__()       

        self.video_capture = None
        self.path_to_labelling_folder = None
        self.paths_to_labels_list = []
        self.path_to_video = None
        self.window_name = None
        self.frame_with_boxes = None
        self.img_rows = None
        self.img_cols = None

        self.screen_width = screen_width
        self.screen_height = screen_height

        # наверное, лучше хранить все рамки в списке, что должно чуть-чуть ускорить обработку
        self.frame_bboxes_list = []

        self.autosave_mode = False

        # список с видимыми рамками. Это костыль, т.к. QListWidget почему-то не сохраняет выделенными строки
        self.temp_bboxes_list = []
     
        open_file_button = QPushButton("Open video")
        close_video_button = QPushButton("Close video")
        save_file_button = QPushButton("Save labels")
        next_frame_button = QPushButton("Next Frame")
        previous_frame_button = QPushButton("Previous Frame")
        self.autosave_current_checkbox = QCheckBox('Autosave Current Boxes')
        show_all_button = QPushButton('Show all classes')
        hide_all_button = QPushButton('Hide all classes')

        search_first_appearance_button = QPushButton("Search for first appearance")

        # чтение списка классов из json
        with open('settings.json', 'r', encoding='utf-8') as fd:
            self.settings_dict = json.load(fd)

        self.class_names_list = self.settings_dict['classes']
        
        #self.classes_combobox = QComboBox(self)
        
        #self.classes_combobox.addItems(self.class_names_list)

        # список отображаемых рамок
        self.visible_classes_list_widget = QListWidget()

        self.visible_classes_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)

        
        self.frame_display = QLCDNumber()
        go_to_frame_button = QPushButton('Go to Frame')

        self.frame_slider = QSlider(Qt.Horizontal)
        self.reset_slider_display()
        self.frame_slider.valueChanged.connect(self.display_frame_position)

        # присоединение к обработчику события
        close_video_button.clicked.connect(self.close_video)
        open_file_button.clicked.connect(self.open_file)
        #save_file_button.clicked.connect(self.save_labels_to_txt)
        go_to_frame_button.clicked.connect(self.set_frame)
        next_frame_button.clicked.connect(self.next_frame_button_handling)
        previous_frame_button.clicked.connect(self.previous_frame_button_handling)
        self.autosave_current_checkbox.stateChanged.connect(self.autosave_current_checkbox_slot)

        self.visible_classes_list_widget.itemClicked.connect(self.update_visible_boxes_on_click_slot)
        self.visible_classes_list_widget.itemEntered.connect(self.update_visible_boxes_on_selection_slot)

        show_all_button.clicked.connect(self.show_all_button_slot)
        hide_all_button.clicked.connect(self.hide_all_button_slot)
        search_first_appearance_button.clicked.connect(self.search_first_appearance_button_slot)

        #self.classes_combobox.currentTextChanged.connect(self.update_current_box_class_name)

        # действия для строки меню
        open_file = QAction('Open', self)
        open_file.setShortcut('Ctrl+O')
        #openFile.setStatusTip('Open new File')
        open_file.triggered.connect(self.open_file)

        # строка меню
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(open_file)

        # выстраивание разметки приложения
        self.grid = QGridLayout()
        self.file_buttons_layout = QVBoxLayout()
        self.displaying_classes_layout = QVBoxLayout()
        self.horizontal_layout = QHBoxLayout()
        self.control_layout = QVBoxLayout()
        self.prev_next_layout = QHBoxLayout()

        self.prev_next_layout.addWidget(previous_frame_button)
        self.prev_next_layout.addWidget(next_frame_button)

        self.control_layout.addWidget(self.frame_display)
        self.control_layout.addWidget(self.frame_slider)
        #self.control_layout.addWidget(self.autosave_current_checkbox)
        self.control_layout.addWidget(go_to_frame_button)
        self.control_layout.addLayout(self.prev_next_layout)

        #self.file_buttons_layout.addWidget(open_file_button)
        #self.file_buttons_layout.addWidget(close_video_button)
        #self.file_buttons_layout.addWidget(save_file_button)

        # пока что спрячем разворачивающийся список классов...
        #self.displaying_classes_layout.addWidget(self.classes_combobox)
        self.displaying_classes_layout.addWidget(self.visible_classes_list_widget)
        self.displaying_classes_layout.addWidget(show_all_button)
        self.displaying_classes_layout.addWidget(hide_all_button)
        #self.displaying_classes_layout.addWidget(search_first_appearance_button)

        #self.horizontal_layout.addLayout(self.file_buttons_layout)
        self.horizontal_layout.addLayout(self.control_layout)
        self.horizontal_layout.addLayout(self.displaying_classes_layout)

        self.main_widget = QWidget()
        self.main_widget.setLayout(self.horizontal_layout)
        self.setCentralWidget(self.main_widget)

        self.setWindowTitle('Video Label Editor')
        
        # Инициализируем поток для показа видео с подключением слотов к сигналам потока
        self.setup_imshow_thread()
        self.show()

    def set_frame(self):
        if self.video_capture is None:
            return
        
        set_frame_dialog = SetFrameDialog(self.frame_number)
        set_frame_dialog.exec()
        if set_frame_dialog.frame_idx is None:
            return
        
        self.current_frame_idx = set_frame_dialog.frame_idx

        self.read_frame()


    def show_info_message_box(self, window_title, info_text):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle(window_title)
        msg_box.setText(info_text)
        msg_box.setStandardButtons(QMessageBox.Ok)
        return msg_box.exec()

    def search_first_appearance_button_slot(self):
        # Сначала надо проверить, что выделен лишь один класс
        qlist_len = self.visible_classes_list_widget.count()
        if qlist_len == 0:
            return

        selected_cnt = 0
        searching_class_name = None
        for item_idx in range(qlist_len):
            if self.visible_classes_list_widget.item(item_idx).isSelected():
                selected_cnt += 1
                searching_class_name = self.visible_classes_list_widget.item(item_idx).data(0)
            if selected_cnt > 1:
                self.show_info_message_box(window_title="Class search info", info_text="You should select only one class for searching")
                return

        if searching_class_name is None:
            self.show_info_message_box(window_title="Class search info", info_text="No class is selected")
            return
        
        for frame_idx, path in enumerate(self.paths_to_labels_list):
            with open(path, 'r') as fd:
                text = fd.read()
            if len(text) == 0:
                return
            
            text = text.split('\n')
            for str_bbox in text:
                try:
                    class_name, x0, y0, x1, y1 = str_bbox.split(',')
                except Exception:
                    continue
                if class_name == searching_class_name:
                    self.current_frame_idx = frame_idx
                    self.read_frame()
                    self.show_info_message_box(
                        window_title="Class search info",
                        info_text=f"First appearance of {class_name} at frame #{frame_idx}")
                    return
        self.show_info_message_box(
            window_title="Class search info",
            info_text=f"{searching_class_name} is not presented")

    def show_all_button_slot(self):
        self.show_or_hide(is_selected=True)
    
    def hide_all_button_slot(self):    
        self.show_or_hide(is_selected=False)

    def show_or_hide(self, is_selected):
        # определяем количество элементов в списке
        qlist_len = self.visible_classes_list_widget.count()
        for item_idx in range(qlist_len):
            class_name = self.visible_classes_list_widget.item(item_idx).data(0)
            for bbox_name, bbox in self.frame_with_boxes.bboxes_dict.items():
                if bbox_name == class_name:
                    self.frame_with_boxes.bboxes_dict[bbox_name].is_visible = is_selected
                    self.visible_classes_list_widget.item(item_idx).setSelected(is_selected)
                    break


    def display_frame_position(self, current_frame_idx):
        if self.video_capture is None or self.frame_with_boxes is None:
            if self.imshow_thread.isRunning():
                self.stop_imshow_thread()
            return
        
        self.frame_display.display(current_frame_idx)
        self.current_frame_idx = current_frame_idx
        self.read_frame()

    def autosave_current_checkbox_slot(self):
        '''
        Обработчик checkbox, отвечающего за автоматическое сохранение кадра при переходе на новый
        '''
        self.autosave_mode = self.autosave_current_checkbox.isChecked()
    
    def setup_imshow_thread(self):
        '''
        При инициализации нового потока необходимо также заново подключать все сигналы
        класса потока к соответствующим слотам главного потока
        '''
        self.imshow_thread = ImshowThread()
        self.imshow_thread.bboxes_update_signal.connect(self.update_visible_classes_list)


    def update_visible_boxes_on_click_slot(self, item):
        '''
        Обновление видимых рамок в кадре. Контролируется посредством visible_classes_list_widget.
        Если элемент выделен, то он отображается в кадре.
        '''
        bbox_name = item.data(0)
        #class_name, bbox_idx = class_str.split(',')
        #print('UPDATE VISIBLE BOXES ON CLICK')
        #print(f'CLASS STR {bbox_name}')
        #bbox_idx = int(bbox_idx)
        if bbox_name in self.frame_with_boxes.bboxes_dict:
            self.frame_with_boxes.bboxes_dict[bbox_name].is_visible = item.isSelected()
            
            '''
            if self.frame_with_boxes is not None:
                for bbox_name, bbox in self.frame_with_boxes.bboxes_dict.items():
                    bbox_class_name = bbox.class_name
                    sample_idx = bbox.id 
                    if bbox_class_name == bbox_name and bbox_idx == sample_idx:
                        bbox.is_visible = item.isSelected()
            '''

        if item.isSelected():
            self.update_current_box_class_name(bbox_name)
        self.update_visible_classes_list()

    
    def update_visible_boxes_on_selection_slot(self, item):
        '''
        Обновление видимых рамок в кадре. Контролируется посредством visible_classes_list_widget.
        Если элемент выделен, то он отображается в кадре.
        '''
        bbox_name = item.data(0)
        #class_name, bbox_idx = class_str.split(',')
        #print('UPDATE VISIBLE BOXES ON CLICK')
        #print(f'CLASS STR {bbox_name}')
        #bbox_idx = int(bbox_idx)
        if bbox_name in self.frame_with_boxes.bboxes_dict:
            self.frame_with_boxes.bboxes_dict[bbox_name].is_visible = item.isSelected()

            self.update_visible_classes_list()
        '''
        class_str = item.data(0)
        class_name, bbox_idx = class_str.split(',')
        bbox_idx = int(bbox_idx)
        if self.frame_with_boxes is not None:
            for bbox_name, bbox in self.frame_with_boxes.bboxes_dict.items():
                bbox_class_name = bbox.class_name 
                sample_idx = bbox.id
                if bbox_class_name == class_name and bbox_idx == sample_idx:
                    bbox.is_visible = item.isSelected()
        '''
        return


    def load_labels_from_file(self):
        '''
        Загружаем из txt-файлов координаты рамок и информацию о классах. 
        Информация загружается в self.frame_with_boxes, 
        self.visible_classes_list_widget не изменяется
        '''
        path_to_to_loading_labels = os.path.join(self.path_to_labelling_folder, f'{self.current_frame_idx:06d}.json')
        
        if os.path.isfile(path_to_to_loading_labels):
            with open(path_to_to_loading_labels, 'r', encoding='utf-8') as fd:
                bboxes_dict = json.load(fd)

            new_bboxes_dict = {}
            for bbox_name, (x0,y0,x1,y1) in bboxes_dict.items():
                class_name,id = bbox_name.split(',')
                new_bboxes_dict[bbox_name] = Bbox(x0, y0, x1, y1, self.img_rows, self.img_cols, class_name, (0,255,0), id, True)

            self.frame_with_boxes.bboxes_dict = new_bboxes_dict
            
            
    
    def update_visible_classes_list(self):
        '''
        обновление списка рамок. 
        Информация о рамках берется из списка рамок, хранящегося в self.frame_with_boxes
        '''

        # определяем количество элементов в списке
        qlist_len = self.visible_classes_list_widget.count()

        #new_list = []
        for bbox_name, bbox in self.frame_with_boxes.bboxes_dict.items():
            #class_name = bbox.class_name
            #sample_idx = bbox.id
            is_selected = bbox.is_visible

            #displayed_name = f'{class_name},{sample_idx}'
            #item = QListWidgetItem(displayed_name)
            
            for item_idx in range(qlist_len):
                item = self.visible_classes_list_widget.item(item_idx)

                class_name_in_list = item.data(0)
                if class_name_in_list == bbox_name:
                    #self.visible_classes_list_widget.item(item_idx).setSelected(bbox.is_visible)
                    self.frame_with_boxes.bboxes_dict[bbox_name].is_visible = item.isSelected()
                    #self.visible_classes_list_widget.item(item_idx).setSelected(bbox.is_visible)

    
            
    def update_current_box_class_name(self, class_name):
        if self.frame_with_boxes is not None:
            self.frame_with_boxes.update_current_class_name(class_name)

    def reset_slider_display(self):
        '''
        Обнуление значений на экране и на слайдере
        '''
        self.frame_slider.setRange(0, 0)
        self.set_slider_display_value(0)

    def set_slider_display_value(self, val):
        self.frame_slider.setValue(val)
        self.frame_display.display(val)

    def setup_slider_range(self, max_val, current_idx):
        '''
        Установка диапазона значений слайдера
        '''
        self.frame_slider.setRange(0, max_val)
        self.set_slider_display_value(current_idx)
        

    def close_video(self):
        '''
        Обработчик закрытия файла
        Сохраняет рамки, закрывает поток чтения изображения и делает объект изображения с рамками пустым
        '''

        if self.frame_with_boxes is not None:
            pass
            #self.save_labels_to_txt()
        self.close_imshow_thread()
        self.frame_with_boxes = None
        self.reset_slider_display()

    def keyPressEvent(self, event):
        if event.text() == '.' or event.text().lower() == 'ю':
            self.next_frame_button_handling()
            return
        elif event.text() == ',' or event.text().lower() == 'б':
            self.previous_frame_button_handling()
            return

        
    def open_file(self):
        self.close_imshow_thread()
        # обнуляем список классов в видео, когда загружаем новое
        self.visible_classes_list_widget.clear()
        # получаем абсолютный путь до файла
        title = 'Open video'

        # записываем в файл settings.json путь до папки с последним открытым файлом
        try:
            last_opened_folder_path = self.settings_dict['last_opened_folder']
        except KeyError:
            last_opened_folder_path = '/home'

        
        # фильтр разрешений файлов
        file_filter = 'Videos (*.mp4 *.wmw *.avi *.mpeg)'
        open_status_tuple = QFileDialog.getOpenFileName(self, title, last_opened_folder_path, file_filter)
        path = open_status_tuple[0]
        if len(path) == 0:
            return

        path = os.sep.join(path.split('/'))
        path_to_folder, name = os.path.split(path)   

        self.settings_dict['last_opened_folder'] = path_to_folder
        with open('settings.json', 'w', encoding='utf-8') as fd:
            json.dump(self.settings_dict, fd)

        label_folder_name = '.'.join(name.split('.')[:-1]) + '_labels'
        
        self.path_to_labelling_folder = os.path.join(path_to_folder, label_folder_name)

        if os.path.isdir(self.path_to_labelling_folder):
            # список путей до txt файлов с координатами рамок номер кадра совпадает с именем файла
            self.paths_to_labels_list = glob.glob(os.path.join(self.path_to_labelling_folder, '*.json'))
        # потом надо изменить!
        else:
            self.paths_to_labels_list = []
            os.mkdir(self.path_to_labelling_folder)

        # заполняем множество имен классов, встретившихся в рамках для отображения в списке
        self.all_classes_set = set()
        for p in self.paths_to_labels_list:
            with open(p, encoding='utf-8') as fd:
                classes_dict = json.load(fd)
            self.all_classes_set.update(classes_dict.keys())

        # инициализируем список классов
        for idx, class_name in enumerate(self.all_classes_set):
            item = QListWidgetItem(class_name)
            self.visible_classes_list_widget.addItem(item)
            self.visible_classes_list_widget.item(idx).setSelected(True)



        # открытие файла
        self.video_capture = cv2.VideoCapture(path)
        # и чтение кадра
        ret, frame = self.video_capture.read()
        if not ret:
            raise RuntimeError(f'Can not read {path} video')
        
        # получение доп. параметров -количесва кадров и размера кадра
        self.frame_number = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.img_rows, self.img_cols = frame.shape[:2]

        if len(self.paths_to_labels_list) > 0:
            self.current_frame_idx = len(self.paths_to_labels_list) - 1
        else:
            self.current_frame_idx = 0
        
        self.current_frame_idx = 0

        # обнуление слайдера управления позицией кадров
        self.setup_slider_range(max_val=self.frame_number, current_idx=self.current_frame_idx)

        self.window_name = name
        
        # инициализация первого кадра
        # ЗОЧЕМ? Разве не лучше
        #self.frame_with_boxes = BboxFrame(img=frame, class_names_list=self.class_names_list, current_class_name=self.class_names_list[0])
        self.frame_with_boxes = BboxFrameTracker(img=frame)

        # инициализация потока, отвечающего за показ кадров
        self.setup_imshow_thread()

        # инициализация карда, котрый мы будем показывать        
        self.imshow_thread.setup_frame(self.frame_with_boxes, self.window_name)
        
        # запуск потока отображения кадра
        self.imshow_thread.start()

        # сразу открываем видео
        self.read_frame()

    def close_imshow_thread(self):
        if self.imshow_thread.isRunning():
            self.frame_with_boxes.delete_img()            
            self.imshow_thread.wait()

    def save_labels(self):
        '''
        Сохранение координат рамок и классов в txt-файл, имя которого совпадает с номером кадра
        СОХРАНЕНИЕ ВЫПОЛНЯЕТСЯ АВТОМАТИЧЕСКИ ПРИ ПЕРЕХОДЕ НА СЛЕДУЮЩИЙ КАДР.
        '''
        path_to_target_json_label = os.path.join(
            self.path_to_labelling_folder, f'{self.current_frame_idx:06d}.json')
        '''        

        '''
        labels_json_dict = {}
        
        # обновляем словарь новыми рамками
        labels_json_dict.update(
            {bbox_name:[int(coord) for coord in bbox.coords] for bbox_name, bbox in self.frame_with_boxes.bboxes_dict.items()})
        
        # Сохраняем разметку
        with open(path_to_target_json_label, 'w', encoding='utf-8') as fd:
            json.dump(labels_json_dict, fd, indent=4)
        
        '''
        if self.frame_with_boxes is not None:
            bboxes = []
            for bbox in self.frame_with_boxes.bboxes_list:
                x0,y0,x1,y1 = bbox.coords
                class_name = bbox.class_info_dict['class_name']
                bboxes.append(f'{class_name},{x0},{y0},{x1},{y1}')

            bboxes = '\n'.join(bboxes)

            path_to_to_saving_labels = os.path.join(self.path_to_labelling_folder, '{:07d}.txt'.format(self.current_frame_idx))
            with open(path_to_to_saving_labels, 'w') as fd:
                fd.write(bboxes)
        '''


    def previous_frame_button_handling(self):
        if self.video_capture is None or self.frame_with_boxes is None:
            if self.imshow_thread.isRunning():
                self.stop_imshow_thread()
            return
        #print('BEFORE SAVE')
        #print(self.frame_with_boxes.bboxes_dict)
        self.save_labels()
        self.current_frame_idx -= 1
        if self.current_frame_idx < 0:
            return
        self.read_frame()


    def next_frame_button_handling(self):
        if self.video_capture is None or self.frame_with_boxes is None:
            if self.imshow_thread.isRunning():
                self.stop_imshow_thread()
            return
        
        # сохраняем все рамки
        if self.current_frame_idx > -1:
            if self.autosave_mode:
                pass
                #self.save_labels_to_txt()
        #print('BEFORE SAVE')
        #print(self.frame_with_boxes.bboxes_dict)
        self.save_labels()
        self.current_frame_idx += 1
        if self.current_frame_idx >= self.frame_number:
            return
        self.read_frame()


    def read_frame(self):
        '''Чтение кадра видео'''
        if self.video_capture is None or self.current_frame_idx >= self.frame_number:
            return
        if self.current_frame_idx < 0:
            self.current_frame_idx = 0
            return
        
        # устанавливаем слайдер кадров в текущее положение
        self.set_slider_display_value(self.current_frame_idx)
        
        # устанавливаем позицию позицию текущего кадра.
        # это необходимо, т.к. мы можем двигаться как назад, так и вперед
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)

        ret, frame = self.video_capture.read()
        '''# Масштабирование кадра
        if self.img_cols / self.screen_width > 0.65 or self.img_rows / self.screen_height > 0.65:
            scaling_factor = 0.65*self.screen_width/self.img_cols
            new_size = tuple(map(lambda x: int(scaling_factor*x), (self.img_cols, self.img_rows)))
            frame = cv2.resize(frame, new_size)
        '''
        if ret:
            # обновляем кадр в потоке, отображающем кадр
            self.frame_with_boxes.update_img(frame)
            # загружаем рамки из файлов 
            self.load_labels_from_file()
            # обновляем список видимых кадров
            self.update_visible_classes_list()
            

    def stop_showing(self):
        if self.is_showing:
            self.is_showing = False
            cv2.destroyAllWindows()


if __name__ == '__main__':
    #s = r'ccf.nrfrtm-.ttrff.mp4'

    app = QApplication(sys.argv)
    screen_resolution = app.desktop().screenGeometry()
    ex = LabelViewerWindow(screen_resolution.width(), screen_resolution.height())
    
    sys.exit(app.exec_())