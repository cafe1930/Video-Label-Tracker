import sys
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QMutex, QObject
from PyQt5.QtWidgets import (QWidget, QComboBox, QPushButton, QTableWidgetItem, QLabel, QCheckBox, QAction, QDialog, QDialogButtonBox,
                             QTableWidget, QLCDNumber, QSlider, QListWidget, QAbstractItemView, QMessageBox, QLineEdit,
                             QHBoxLayout, QFileDialog, QVBoxLayout, QApplication, QMainWindow, QGridLayout, QListWidgetItem)

from PyQt5.QtGui import QKeyEvent

import cv2
import os
import glob
import json

from copy import deepcopy

import numpy as np

import time
import torch

from opencv_frames import BboxFrame, BboxFrameTracker, Bbox, process_box_coords, xywh2xyxy, compute_bbox_area

from ultralytics import YOLO

def show_info_message_box(window_title, info_text, buttons, icon_type):
    msg_box = QMessageBox()
    msg_box.setIcon(icon_type)
    msg_box.setWindowTitle(window_title)
    msg_box.setText(info_text)
    msg_box.setStandardButtons(buttons)
    return msg_box.exec()

class One2OneMapping:
    def __init__(self, mapping={}):
        self.inverse_mapping_dict = {v: k for k, v in mapping.items()}
        self.forward_mapping_dict = {v: k for k, v in self.inverse_mapping_dict.items()}
        

    def forward_mapping(self, key):
        return self.forward_mapping_dict[key]
    
    def inverse_mapping(self, key):
        return self.inverse_mapping_dict[key]
    
    def update(self, forward_mapping_dict):
        inverse_mapping_dict = {v: k for k, v in forward_mapping_dict.items()}
        self.inverse_mapping_dict.update(inverse_mapping_dict)
        self.forward_mapping_dict = {v: k for k, v in self.inverse_mapping_dict.items()}
        self.inverse_mapping_dict = {v: k for k, v in self.forward_mapping_dict.items()}
    
    def __repr__(self):
        return f'forward map: {self.forward_mapping_dict}, inverse map: {self.inverse_mapping_dict}'



class LabelNewBoxDialog(QDialog):
    def __init__(self, current_bbox, obj_descr2registered_bbox_dict, tracked_and_raw_bboxes_dict):
        '''
        Данное диалоговое окно служит для выбора человека, движение которого мы отслеживаем
        Диалоговое окно должно выскакивать после первого кадра.
        '''
        super().__init__()
        
        self.setMinimumWidth(250)

        # мы изменяем рамку, которая была добавлена последней
        self.current_bbox = deepcopy(current_bbox)
        
        # словарь, выполняющий отображение описание людей на "синие" (зарегистрированные) рамки
        # ключ - словесное описание объекта, значение - имя отображаемой камки
        self.obj_descr2registered_bbox_dict = obj_descr2registered_bbox_dict

        self.tracked_and_raw_bboxes_dict = tracked_and_raw_bboxes_dict

        # показатели, которые мы меняем
        self.current_obj_descr = None

        self.confirm_bbox_creation = False


        self.setWindowTitle('Присваивание имени новой рамке')
        self.confirm_button = QPushButton('Сохранить')
        self.cancell_button = QPushButton('Отменить')
        self.existent_classes_combobox = QComboBox()
        self.class_idx_text_label = QLabel()
        
        # заполнение выпадающих списков значениями
        self.existent_classes_combobox.addItems(list(self.obj_descr2registered_bbox_dict.keys()))
        self.existent_classes_combobox.activated[str].connect(self.combobox_value_changed)
        self.confirm_button.clicked.connect(self.confirm_and_exit)
        self.cancell_button.clicked.connect(self.cancell_and_exit)

        self.current_obj_descr = self.existent_classes_combobox.currentText()
        self.current_bbox_name = self.obj_descr2registered_bbox_dict[self.current_obj_descr]

        functional_layout = QHBoxLayout()
        functional_layout.addWidget(self.existent_classes_combobox)
        functional_layout.addWidget(self.class_idx_text_label)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.confirm_button)
        buttons_layout.addWidget(self.cancell_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(functional_layout)
        main_layout.addLayout(buttons_layout)
        self.setLayout(main_layout)

    def confirm_and_exit(self):
        '''
        Меняем имя класса и номер у рамки
        '''
        if self.current_bbox_name in self.tracked_and_raw_bboxes_dict:
            result = show_info_message_box(
                window_title='Коллизия имен рамок',
                info_text=f'Рамка {self.current_bbox_name} для объекта {self.current_obj_descr} уже есть на кадре. Перезаписать ее?',
                buttons=QMessageBox.Yes | QMessageBox.No,
                icon_type=QMessageBox.Warning
            )
            if result == QMessageBox.No:
                return
        self.confirm_bbox_creation = True
        class_name, id = self.current_bbox_name.split(',')
        
        self.current_bbox.class_name = class_name
        self.current_bbox.id = id
        self.current_bbox.color = (0, 255, 0)
        self.current_bbox.is_additionaly_tracked = True
        
        self.close()

    def cancell_and_exit(self):
        self.confirm_bbox_creation = False
        self.close()

    def combobox_value_changed(self, value):
        # ищем рамку с теми же самыми именем класса и индексом
        self.current_obj_descr = value
        self.current_bbox_name = self.obj_descr2registered_bbox_dict[value]


class RegisterPersonsDialog(QDialog):
    '''
    '''
    def __init__(self, raw_bbox2registered_bbox_mapping,  obj_descr2registered_bbox_dict, raw_bboxes_dict):
        '''
        Данное диалоговое окно служит для выбора человека, движение которого мы отслеживаем
        Диалоговое окно должно выскакивать после первого кадра.
        '''
        super().__init__()
        
        self.setMinimumWidth(250)

        # словарь для отображения имени отслеживаемого объекта на имя "черной" (не обработанной) рамки
        # {имя автоматически сгенерированной рамки: описание объекта}
        #self.obj_descr2raw_bbox_dict = {}
        self.raw_bbox2registered_bbox_mapping = raw_bbox2registered_bbox_mapping

        # показвтели, которые мы меняем
        self.current_raw_bbox_name = None
        self.current_obj_descr = None
        
        # {описание отслеживаемого объекта: имя зарегистрированной рамки}
        self.obj_descr2registered_bbox_dict = obj_descr2registered_bbox_dict

        # этот словарь нужен для индексации экземпляров отдельных классов
        #self.classes_ids_dict = self.get_all_classes_ids_dict()

        self.setWindowTitle('Ассоциация имен классов с рамками')
        
        self.save_and_exit_button = QPushButton('Сохранить и выйти')
        self.exit_without_save_button = QPushButton('Выйти без сохранения')
        self.class_names_combobox = QComboBox()
        self.bboxes_combobox = QComboBox()
        
        
        # заполнение выпадающих списков значениями
        self.class_names_combobox.addItems(list(self.obj_descr2registered_bbox_dict.keys()))
        self.bboxes_combobox.addItems(list(raw_bboxes_dict.keys()))
        
        self.class_names_combobox.activated[str].connect(self.class_names_combobox_value_changed)
        self.bboxes_combobox.activated[str].connect(self.bboxes_combobox_value_changed)

        self.save_and_exit_button.clicked.connect(self.save_and_exit)
        self.exit_without_save_button.clicked.connect(self.exit_without_save)
        

        functional_layout = QHBoxLayout()
        functional_layout.addWidget(self.class_names_combobox)
        functional_layout.addWidget(self.bboxes_combobox)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.save_and_exit_button)
        buttons_layout.addWidget(self.exit_without_save_button)
                

        main_layout = QVBoxLayout()
        main_layout.addLayout(functional_layout)
        main_layout.addLayout(buttons_layout)
        self.setLayout(main_layout)

    def exit_without_save(self):
        self.close()

    def save_and_exit(self):
        '''
        Меняем имя класса и номер у рамки
        '''
        if self.current_raw_bbox_name is None:
            show_info_message_box('Нет изменений', 'Не выбрана рамка', QMessageBox.Ok, QMessageBox.Critical)
            return
        if self.current_obj_descr is None:
            show_info_message_box('Нет изменений', 'Не выбрано имя класса', QMessageBox.Ok, QMessageBox.Critical)
            return
        registered_bbox_name = self.obj_descr2registered_bbox_dict[self.current_obj_descr]
        self.raw_bbox2registered_bbox_mapping.update({self.current_raw_bbox_name: registered_bbox_name})
        self.close()

    def bboxes_combobox_value_changed(self, value):
        # ищем рамку с теми же самыми именем класса и индексом
        #class_name, id = value.split(',')
        #!!!!
        self.current_raw_bbox_name = value
        #self.current_bbox.class_name = class_name

    def class_names_combobox_value_changed(self, value):
        self.current_obj_descr = value

class SetFrameDialog(QDialog):
    def __init__(self, frames_num):
        super().__init__()
        self.frames_num = frames_num
        self.frame_idx = None

        self.text_line = QLineEdit()
        self.save_and_exit_button = QPushButton('Сохранить и выйти')
        self.exit_without_save_button = QPushButton('Выйти без сохранения')
        self.text_line.textChanged[str].connect(self.text_line_handling)

        self.save_and_exit_button.clicked.connect(self.save_and_exit)
        self.exit_without_save_button.clicked.connect(self.exit_without_save)
        

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.text_line)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.save_and_exit_button)
        buttons_layout.addWidget(self.exit_without_save_button)
        main_layout.addLayout(buttons_layout)
        self.setLayout(main_layout)

    def save_and_exit(self):
        if self.frame_idx is None:
            show_info_message_box(
                'ВНИМАНИЕ!',
                'Номер кадра не выбран, поэтому переход на другой кадр не будет осуществлен',
                QMessageBox.Ok,
                QMessageBox.Warning
            )
        self.close()

    def exit_without_save(self):
        self.frame_idx = None
        self.close()

    def text_line_handling(self, text):
        try:
            self.frame_idx = int(text)
        except:
            show_info_message_box(
                'ОШИБКА!',
                'Необходимо вводить цифры!',
                QMessageBox.Ok,
                QMessageBox.Critical
            )
            self.text_line.setText('')
            return
        
        if self.frame_idx >= self.frames_num:
            self.frame_idx = None
            show_info_message_box(
                'ОШИБКА!',
                f'Номер кадра, на который планируется переход ({self.frame_idx}), не должен превышать общее количество ({self.frames_num})!',
                QMessageBox.Ok,
                QMessageBox.Critical
            )
            self.text_line.setText('')
            return
        
class SelectDetector(QDialog):
    def __init__(self):
        super().__init__()
        detectors_names_list = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
        self.combobox = QComboBox()
        self.combobox.addItems(detectors_names_list)
        self.combobox.activated[str].connect(self.select_detector)
        self.current_detector = 'yolov8x.pt' if torch.cuda.is_available() else 'yolov8s'

        self.buttons = QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.combobox)
        self.main_layout.addWidget(self.buttons)
        self.setLayout(self.main_layout)

    def select_detector(self, val):
        self.current_detector = f'{val}.pt'

class TrackerWindow(QMainWindow):
    def __init__(self, screen_width, screen_height, tracker_type='yolov8x.pt'):
        super().__init__()       

        self.set_params_to_default()

        self.tracker_type = 'yolov8x.pt' if torch.cuda.is_available() else 'yolov8s.pt'

        self.tracker = None #Yolov8Tracker(model_type=tracker_type) он инициализируется приоткрытии файла

        self.screen_width = screen_width
        self.screen_height = screen_height

        bboxes_correction_info_str = \
        '''Справка по коррекции рамок:

Перемещение: зажатый Ctrl+зажатая ЛКМ внутри рамки

Коррекция углов: зажатый Ctrl+зажатая ЛКМ в углу рамки

Удаление рамки: зажатый Alt+клик ЛКМ внутри рамки

'''
        # флаг, разрешающий автоматическое воспроизведение видео
        
        bboxes_correction_info = QLabel(text=bboxes_correction_info_str)
        next_frame_button = QPushButton("След. кадр >")
        previous_frame_button = QPushButton("< Пред. кадр")
        autoplay_button = QPushButton("Autoplay 30 frames")
        self.autosave_current_checkbox = QCheckBox('Autosave Current Boxes')
        self.show_tracked_checkbox = QCheckBox('Показать отслеживаемые объекты')

        register_objects_button = QPushButton('Регистрация нового объекта')
        cancell_register_objects_button = QPushButton('Отмена регистрации всех объектов')
        
        reset_tracker_and_set_frame_button = QPushButton('Сбросить трекер и перейти на заданный кадр')
        
        show_raw_button = QPushButton('Показать только автоматически сгенерированные рамки')
        show_registered_button = QPushButton('Показать все зарегистрированные рамки')
        show_tracked_button = QPushButton('Показать все отслеживаемые рамки')
        show_tracked_and_raw_button = QPushButton('Показать отслеживаемые и сгенерированные рамки')
        #show_tracking_button = QPushButton('Показать отслеживаемые рамки')

        #self.enable_tracking_checkbox = QCheckBox('Enable automatic tracking')

        self.classes_with_description_table = QTableWidget()
        self.classes_with_description_table.setColumnCount(2)
        #self.classes_with_description_table.setW
        self.classes_with_description_table.setHorizontalHeaderLabels(["Описание объекта", "Обозначение\nрамки"])
        self.classes_with_description_table.setColumnWidth(0, 170)
        self.classes_with_description_table.setColumnWidth(1, 80)
        self.classes_with_description_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        
        #search_first_appearance_button = QPushButton("Search for first appearance")

        # чтение списка классов из json
        with open('settings.json', 'r', encoding='utf-8') as fd:
            self.settings_dict = json.load(fd)

        self.class_names_list = self.settings_dict['classes']
        
        #self.classes_combobox = QComboBox(self)
        #self.classes_combobox.addItems(self.class_names_list)

        # список отображаемых рамок
        #self.visible_classes_list_widget = QListWidget()
        #self.visible_classes_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)

        self.current_frame_label = QLabel()
        self.current_frame_label.setText('Идекс текущего кадра:')
        self.current_frame_display = QLCDNumber()
        self.frames_num_label = QLabel()
        self.frames_num_label.setText('Общее количество кадров:')
        self.all_frames_display = QLCDNumber()
        #self.frame_slider = QSlider(Qt.Horizontal)
        self.reset_display()
        #self.frame_slider.valueChanged.connect(self.display_frame_position)

        # присоединение к обработчику события
        next_frame_button.clicked.connect(self.next_frame_button_handling)
        previous_frame_button.clicked.connect(self.previous_frame_button_handling)
        autoplay_button.clicked.connect(self.autoplay)
        register_objects_button.clicked.connect(self.register_persons_handling)
        cancell_register_objects_button.clicked.connect(self.cancell_register_objects_button_handling)
        reset_tracker_and_set_frame_button.clicked.connect(self.reset_tracker_and_set_frame_button_handling)
        show_raw_button.clicked.connect(self.show_raw_button_handling)
        show_registered_button.clicked.connect(self.show_registered_button_handing)
        show_tracked_button.clicked.connect(self.show_tracked_button_handling)
        show_tracked_and_raw_button.clicked.connect(self.show_tracked_and_raw_button_handling)
        self.autosave_current_checkbox.stateChanged.connect(self.autosave_current_checkbox_slot)
        self.show_tracked_checkbox.stateChanged.connect(self.show_tracked_checkbox_slot)

        #self.classes_combobox.currentTextChanged.connect(self.update_current_box_class_name)
        self.classes_with_description_table.cellClicked.connect(self.table_cell_click_handling)
        self.classes_with_description_table.cellEntered.connect(self.table_cell_click_handling)

        # действия для строки меню
        open_file = QAction('Open', self)
        open_file.setShortcut('Ctrl+O')
        #openFile.setStatusTip('Open new File')
        open_file.triggered.connect(self.open_file)

        change_detector = QAction('Change Detector Type', self)
        change_detector.triggered.connect(self.change_detector)
       
        # строка меню
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        editMenu = menubar.addMenu('&Edit')
        fileMenu.addAction(open_file)
        editMenu.addAction(change_detector)
        

        # выстраивание разметки приложения
        self.grid = QGridLayout()
        
        self.displaying_classes_layout = QVBoxLayout()
        self.horizontal_layout = QHBoxLayout()
        self.bboxes_display_control_layout = QVBoxLayout()
        self.control_layout = QVBoxLayout()
        self.prev_next_layout = QHBoxLayout()

        self.bboxes_display_control_layout.addWidget(bboxes_correction_info)
        self.bboxes_display_control_layout.addWidget(show_raw_button)
        self.bboxes_display_control_layout.addWidget(show_registered_button)
        self.bboxes_display_control_layout.addWidget(show_tracked_button)
        self.bboxes_display_control_layout.addWidget(show_tracked_and_raw_button)
        #self.bboxes_display_control_layout.addWidget(reset_tracker_and_set_frame_button)
        self.bboxes_display_control_layout.addStretch(1)

        self.prev_next_layout.addWidget(previous_frame_button)
        self.prev_next_layout.addWidget(next_frame_button)

        #self.control_layout.addWidget(self.enable_tracking_checkbox)
        self.control_layout.addWidget(self.current_frame_label)
        self.control_layout.addWidget(self.current_frame_display)
        self.control_layout.addWidget(self.frames_num_label)
        self.control_layout.addWidget(self.all_frames_display)
        self.control_layout.addWidget(reset_tracker_and_set_frame_button)

        #self.control_layout.addWidget(self.frame_slider)
        #self.control_layout.addWidget(self.autosave_current_checkbox)
        self.control_layout.addWidget(autoplay_button)
        self.control_layout.addLayout(self.prev_next_layout)
        #

        # пока что спрячем разворачивающийся список классов...
        self.displaying_classes_layout.addWidget(self.classes_with_description_table)

        self.displaying_classes_layout.addWidget(register_objects_button)
        self.displaying_classes_layout.addWidget(cancell_register_objects_button)
        
        self.horizontal_layout.addLayout(self.bboxes_display_control_layout)
        self.horizontal_layout.addLayout(self.displaying_classes_layout)
        self.horizontal_layout.addLayout(self.control_layout)

        self.main_widget = QWidget()
        self.main_widget.setLayout(self.horizontal_layout)
        self.setCentralWidget(self.main_widget)

        self.setWindowTitle('Video Label Tracker')
        
        # Инициализируем поток для показа видео с подключением слотов к сигналам потока
        self.setup_imshow_thread()
        self.show()

    def set_tracking_params_to_default(self):
        # список, содержащий всё вместе - и отслеживаемые рамки, и автоматически сгенерированные
        # {имя рамки: объект рамки}
        self.tracked_and_raw_bboxes_dict = {}

        # Список для того, чтобы выявлять новые рамки
        self.previous_tracked_and_raw_bboxes_dict = {}
        # словарь для транслирования индексов рамок различных людей на видео
        # {словесные описания объектов: имена отображаемых рамок вида person,1}
        # заполняется при чтении файлов, сопутствующих видео
        self.obj_descr2registered_bbox_dict = {}
        # обратный obj_descr2registered_bbox_dict словарь
        self.registered_bbox2obj_descr_dict = {}

        # словарь для транслирования индексов автоматически сгенерированных рамок на имена отслеживаемых людей 
        # для одного кадра. Содержит однозначные отображения и не допускает исключений, которые нельзя допускать в рамках одного кадра
        # {имя автоматически сгенерированной рамки: имя зарегистрированной рамки}
        self.reistered_raw_bbox_name2registered_bbox_name_mapping = One2OneMapping()
        # доп словарь, нужжный для фильтрации ошибок при регистрации нового человвека
        # {имя автоматически сгенерированной рамки: имя зарегистрированной рамки}
        self.current_frame_raw_bbox_name2registered_bbox_name_mapping = One2OneMapping()

        # словарь, содержащий множественные отображения имен автоматически сгенерированных рамок на описания отслеживаемых объектов
        # {имя автоматически сгенерированной рамки: имя зарегистрированной рамки}
        self.all_frames_raw_bbox_name2registered_bbox_name_dict = {}
        
        # названия рамок объектов, которых мы отслеживаем
        # удобно хранить в множестве (set), для быстрого доступа и извлечения
        self.tracking_bboxes_names_set = set()
        
        self.is_autoplay = False

        # словарь с рамками зарегистрированных объектов, которых мы отслеживаем
        # {имя зарегистрированной рамки: объект рамки}
        self.tracking_bboxes_dict = {}

        # список рамок, которым присвоены названия объектов
        # {имя зарегистрированной рамки: объект рамки}
        self.registered_bboxes_dict = {}

        # список не обработанных рамок, полученных в результате рисования или детекции
        # структура словаря: {имя автоматически сгенерированной рамки: объект рамки}
        self.raw_bboxes_dict = {}

        # флаг для отображения либо не обработанных, либо зарегистрированных рамок
        self.is_tracked_bboxes_show = False


    def set_params_to_default(self):
        self.video_capture = None
        self.path_to_labelling_folder = None
        self.paths_to_labels_list = []
        self.path_to_video = None
        self.window_name = None
        self.frame_with_boxes = None
        self.img_rows = None
        self.img_cols = None

        # наверное, лучше хранить все рамки в списке, что должно чуть-чуть ускорить обработку
        self.all_frames_bboxes_list = []

        self.is_autoplay = False

        self.set_tracking_params_to_default()

        self.autosave_mode = False

        # список с видимыми рамками. Это костыль, т.к. QListWidget почему-то не сохраняет выделенными строки
        self.temp_bboxes_list = []

        #self.reset_table()

    def change_detector(self):
        select_detector_dialog = SelectDetector()
        is_changed = select_detector_dialog.exec()
        self.is_autoplay = False
        if not is_changed:
            return
        else:
            #print(f'{select_detector_dialog.current_detector}')
            self.tracker_type = select_detector_dialog.current_detector
            if self.video_capture is None:
                return
            else:
                ret = show_info_message_box(
                    'ВНИМАНИЕ!',
                    'При смене детектора процесс отслеживания обнуляется и начинается заново! Снимается выделение со всех отслеживаемых объектов, их придется выбирать заново. Выполнить?',
                    QMessageBox.Yes|QMessageBox.No,
                    QMessageBox.Warning
                )
                if ret == QMessageBox.No:
                    return
                else:
                    self.unselect_all_table_items()
                    self.set_tracking_params_to_default()
                    self.registered_bboxes_dict, self.tracking_bboxes_dict, self.tracked_and_raw_bboxes_dict \
                        = self.update_registered_and_tracking_objects_dicts('raw')
                    self.reset_tracker()
                    self.read_persons_description()

                    self.read_frame(direction='forward')



    '''
    def enable_tracking_checkbox_handling(self):
        
        if self.enable_tracking_checkbox.checkState() == 0:
            # надо предупредить о том, что надо сохоранить файлы
            # !!!!!!

            self.tracker = None
        else:
            if self.tracker is None:
                self.tracker = Tracker()
    '''
    def reset_tracker_and_set_frame_button_handling(self):
        # Вызов диалогового окна, куда передается self.frame_number
        #self.frame_number
        #self.
        if self.video_capture is None:
            return
        ret = show_info_message_box(
            'Внимание!',
            'При переходе на заданный кадр процесс отслеживания обнуляется и начинается заново! Снимается выделение со всех отслеживаемых объектов, их придется выбирать заново. Выполнить?',
            QMessageBox.Yes|QMessageBox.No,
            QMessageBox.Warning
            )
        self.is_autoplay = False
        if ret == QMessageBox.No:
            return
        

        set_frame_dialog = SetFrameDialog(self.frame_number)
        set_frame_dialog.exec()
        if set_frame_dialog.frame_idx is None:
            return
        
        self.current_frame_idx = set_frame_dialog.frame_idx

        self.unselect_all_table_items()
        self.set_tracking_params_to_default()
        self.registered_bboxes_dict, self.tracking_bboxes_dict, self.tracked_and_raw_bboxes_dict \
            = self.update_registered_and_tracking_objects_dicts('raw')
        self.reset_tracker()
        self.read_persons_description()

        self.read_frame(direction='forward')

        

    def show_raw_button_handling(self):
        if self.frame_with_boxes is None:
            return
        self.frame_with_boxes.bboxes_dict = self.raw_bboxes_dict

    def show_tracked_and_raw_button_handling(self):
        if self.frame_with_boxes is None:
            return
        self.frame_with_boxes.bboxes_dict = self.tracked_and_raw_bboxes_dict

    def show_tracked_button_handling(self):
        if self.frame_with_boxes is None:
            return
        self.frame_with_boxes.bboxes_dict = self.tracking_bboxes_dict

    def show_registered_button_handing(self):
        if self.frame_with_boxes is None:
            return
        self.frame_with_boxes.bboxes_dict = self.registered_bboxes_dict

    def cancell_register_objects_button_handling(self):
        if len(self.obj_descr2registered_bbox_dict) == 0:
            return
        ret = show_info_message_box(
                'ВНИМАНИЕ!',
                'Данное действие удалит все зарегистрированные объекты. Придется выполнять регистрацию заново и выбирать объект отслеживания',
                QMessageBox.Yes|QMessageBox.No,
                QMessageBox.Warning)
        self.is_autoplay = False
        if ret == QMessageBox.No:
            return
        
        # {имя рамки: объект рамки}
        self.tracked_and_raw_bboxes_dict = {}

        # Список для того, чтобы выявлять новые рамки
        self.previous_tracked_and_raw_bboxes_dict = {}
        
        # {имя автоматически сгенерированной рамки: имя зарегистрированной рамки}
        self.tracked_and_raw_bboxes_dict = self.raw_bboxes_dict
        self.current_frame_raw_bbox_name2registered_bbox_name_mapping = One2OneMapping()

        # словарь, содержащий множественные отображения имен автоматически сгенерированных рамок на описания отслеживаемых объектов
        # {имя автоматически сгенерированной рамки: имя зарегистрированной рамки}
        self.all_frames_raw_bbox_name2registered_bbox_name_dict = {}

        self.registered_bboxes_dict, self.tracking_bboxes_dict, self.tracked_and_raw_bboxes_dict \
            = self.update_registered_and_tracking_objects_dicts('raw')
        self.reset_tracker()

        self.read_frame(direction='forward')


    def register_persons_handling(self):
        self.is_autoplay = False
        if len(self.obj_descr2registered_bbox_dict) == 0:
            show_info_message_box(
                'Ошибка загрузки описаний объектов',
                'Сначала загрузите видео с описением классов',
                QMessageBox.Ok,
                QMessageBox.Critical
                )
            return
        
        if len(self.frame_with_boxes.bboxes_dict) == 0:
            show_info_message_box(
                'Ошибка рамок объектов',
                'На видео не обнаружены объекты',
                QMessageBox.Ok,
                QMessageBox.Critical)
            
            return               
        

        # сначала показываем автоматически сгенерированные рамки
        self.frame_with_boxes.bboxes_dict = self.raw_bboxes_dict

        # вызов диалогового окна позволяет изменить имя класса всего для одной рамки
        register_persons_dialog = RegisterPersonsDialog(
            #self.obj_descr2raw_bbox_dict,
            self.current_frame_raw_bbox_name2registered_bbox_name_mapping,
            self.obj_descr2registered_bbox_dict,
            self.raw_bboxes_dict)
        register_persons_dialog.exec()

        if register_persons_dialog.current_obj_descr is None or register_persons_dialog.current_raw_bbox_name is None:
            return 1
        """
        print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        print('______DEBUG_UPDATING_DICT______')
        print('BEFORE UPDATE')
        print('(self.current_frame_raw_bbox_name2registered_bbox_name_mapping):')
        print(self.current_frame_raw_bbox_name2registered_bbox_name_mapping)
        print('(self.all_frames_raw_bbox_name2registered_bbox_name_dict)')
        print(self.all_frames_raw_bbox_name2registered_bbox_name_dict)
        print()
        """
        
        self.current_frame_raw_bbox_name2registered_bbox_name_mapping = register_persons_dialog.raw_bbox2registered_bbox_mapping
        
        # Вычищаем общий словарь отображений от повторяющихся имен зарегистрированных рамок
        for raw_bbox_name, registered_bbox_name in list(self.all_frames_raw_bbox_name2registered_bbox_name_dict.items()):
            if registered_bbox_name in self.current_frame_raw_bbox_name2registered_bbox_name_mapping.inverse_mapping_dict:
                #key_to_pop = self.current_frame_raw_bbox_name2registered_bbox_name_mapping.inverse_mapping_dict[registered_bbox_name]
                self.all_frames_raw_bbox_name2registered_bbox_name_dict.pop(raw_bbox_name)

        # обновляем словарь возможных отображений {имя сгенерированной рамки: имя зарегистрированной рамки}
        self.all_frames_raw_bbox_name2registered_bbox_name_dict.update(self.current_frame_raw_bbox_name2registered_bbox_name_mapping.forward_mapping_dict)
        """
        print('AFTER UPDATE')
        print('(self.current_frame_raw_bbox_name2registered_bbox_name_mapping):')
        print(self.current_frame_raw_bbox_name2registered_bbox_name_mapping)
        print('(self.all_frames_raw_bbox_name2registered_bbox_name_dict)')
        print(self.all_frames_raw_bbox_name2registered_bbox_name_dict)
        print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        """

        self.register_new_bbox()
        self.registered_bboxes_dict, self.tracking_bboxes_dict, self.tracked_and_raw_bboxes_dict \
            = self.update_registered_and_tracking_objects_dicts('raw') #???
        


    def update_frame_raw_bboxes2registered_bboxes_dict(self, new_raw_bbox2obj_descr_dict):
        '''
        Обновление словаря для перевода имен распознанных рамок в описание объектов для одного кадра
        '''
        for raw_bbox_name, obj_descr in new_raw_bbox2obj_descr_dict.items():
            # обработка коллизий. На одном кадре одна рамка может принадлежать только одному объекту
            if obj_descr in self.current_frame_raw_bbox_name2registered_bbox_name_dict.values():
                # для сохранения однозначности соотнесения между описанием и рамкой на одном кадре
                # надо удалить элемент с тем же сзначением
                key_idx = list(self.current_frame_raw_bbox_name2registered_bbox_name_dict.values()).index(obj_descr)
                bbox_to_remove = list(self.current_frame_raw_bbox_name2registered_bbox_name_dict.keys())[key_idx]
                # жуткий костыль!!! строка '(R)' есть признак того, что рамка сгенерирована автоматически
                if '(R)' not in bbox_to_remove:
                    continue
                removed = self.current_frame_raw_bbox_name2registered_bbox_name_dict.pop(bbox_to_remove, None)
            registered_bbox_name = self.obj_descr2registered_bbox_dict[obj_descr]
            self.current_frame_raw_bbox_name2registered_bbox_name_dict[raw_bbox_name] = registered_bbox_name
    

    def register_new_bbox(self):
        '''
        Обновление списка рамок в соответствии с обновленными сведениями об именах классов
        '''
        # на всякий случай копируем, иначе изменяются вообще все рамки!
        for raw_name, raw_bbox in self.raw_bboxes_dict.items():
            # пока что такой костыль - "зарегистрированная" рамка - это рамка, выкрашенная в определнный цвет
            if raw_bbox.color == (255, 0, 0):
                continue

            raw_bbox_name = f'{raw_bbox.class_name},{raw_bbox.id}'
            try:
                # если имени сгенерированной рамки нет в словаре, отображающем имена сгенерированных  
                # рамок на описание человека, значит эта рамка еще не проассоциирована
                #new_bbox_name = self.current_frame_raw_bbox_name2registered_bbox_name_mapping.forward_mapping(raw_bbox_name)
                new_bbox_name = self.all_frames_raw_bbox_name2registered_bbox_name_dict[raw_bbox_name]
            except KeyError:
                continue

            # имени рамки нет в словаре, значит она еще не проассоциирована
            #new_bbox_name = self.obj_descr2registered_bbox_dict[person_descr]

            new_class_name, new_id = new_bbox_name.split(',')
            new_id = int(new_id)

            x0, y0, x1, y1 = raw_bbox.coords
            bbox = Bbox(
                x0, y0, x1, y1,
                raw_bbox.img_rows,
                raw_bbox.img_cols, 
                new_class_name,
                (255, 0, 0),
                new_id)

            self.registered_bboxes_dict[new_bbox_name] = bbox
  
    
    def keyPressEvent(self, event):
        if event.text() == '.' or event.text().lower() == 'ю':
            self.next_frame_button_handling()
            return
        elif event.text() == ',' or event.text().lower() == 'б':
            self.previous_frame_button_handling()
            return
        
        if event.text().lower() == ']' or event.text().lower() == 'ъ':
            self.autoplay()
            return
    

    def update_registered_and_tracking_objects_dicts(self, update_source):
        '''
        update_type:str - на основании какого списка рамок делается обновление
        '''
        
        # список, содержащий рамки для отображения
        displaying_bboxes_dict = {}
        new_tracking_bboxes_dict = {}
        new_registered_bboxes_dict = {}
        
        if update_source == 'raw':
            # обновляем список отслеживаемых рамок, извлекая отслеживаемые объекты из таблицы
            self.update_tracking_objects_set_from_table()
            # обновляем словари с зарегистрированными рамками и отслеживаемыми рамками
            for raw_bbox_name, raw_bbox in self.raw_bboxes_dict.items():
                try:
                    # если в словаре self.all_frames_raw_bbox_name2registered_bbox_name_dict отсутствует
                    # имя автоматически сгенерированной рамки, то идем дальше
                    corresponding_registered_name = self.all_frames_raw_bbox_name2registered_bbox_name_dict[raw_bbox_name]
                except KeyError:
                    # записываем в словарь отображаемых рамок автоматически сгенерированную рамку
                    displaying_bboxes_dict[raw_bbox_name] = raw_bbox
                    continue
                class_name, id = corresponding_registered_name.split(',')
                id = int(id)
                # проверяем, находится ли текущая рамка в перечне отслеживаемых рамок
                if corresponding_registered_name in self.tracking_bboxes_names_set:
                    tracking_bbox = deepcopy(raw_bbox)
                    tracking_bbox.class_name = class_name
                    tracking_bbox.id = id
                    tracking_bbox.color = (0, 255, 0)
                    new_tracking_bboxes_dict[corresponding_registered_name] = tracking_bbox
                    displaying_bboxes_dict[corresponding_registered_name] = tracking_bbox
                else:
                    displaying_bboxes_dict[raw_bbox_name] = raw_bbox

                registered_bbox = deepcopy(raw_bbox)
                registered_bbox.class_name = class_name
                registered_bbox.id = id
                registered_bbox.color = (255, 0, 0)
                new_registered_bboxes_dict[corresponding_registered_name] = registered_bbox
        
        elif update_source == 'raw_and_tracked':
            self.update_tracking_objects_set_from_drawn_bbox()
            displaying_bboxes_dict = self.tracked_and_raw_bboxes_dict
            for raw_tracked_bbox_name, raw_tracked_bbox in self.tracked_and_raw_bboxes_dict.items():
                class_name, id = raw_tracked_bbox_name.split(',')
                if '(R)' not in raw_tracked_bbox_name:
                    registered_bbox = deepcopy(raw_tracked_bbox)
                    registered_bbox.class_name = class_name
                    registered_bbox.id = id
                    registered_bbox.color = (255, 0, 0)
                    new_registered_bboxes_dict[raw_tracked_bbox_name] = registered_bbox
                
                if raw_tracked_bbox_name in self.tracking_bboxes_names_set:
                    tracking_bbox = deepcopy(raw_tracked_bbox)
                    tracking_bbox.class_name = class_name
                    tracking_bbox.id = id
                    tracking_bbox.color = (0, 255, 0)
                    new_tracking_bboxes_dict[raw_tracked_bbox_name] = tracking_bbox
        else:
            raise ValueError(f'Parameter <update_source> should be equal either to str "raw" or str "raw_and_tracked"')
        
        # сортируем рамки по убыванию имен пока что выглядит как костыль
        displaying_bboxes_dict = dict(sorted(displaying_bboxes_dict.items(), reverse=True))
            
        return new_registered_bboxes_dict, new_tracking_bboxes_dict, displaying_bboxes_dict

    def update_tracking_objects_set_from_drawn_bbox(self):
        rows_num = self.classes_with_description_table.rowCount()
        tracked_bboxes_names = [name for name in self.tracked_and_raw_bboxes_dict.keys() if '(R)' not in name]
        for row_idx in range(rows_num):
            obj_descr_item = self.classes_with_description_table.item(row_idx, 0)
            registered_bbox_name_item = self.classes_with_description_table.item(row_idx, 1)
            registered_bbox_name = registered_bbox_name_item.text()
            
            if obj_descr_item is not None or registered_bbox_name_item is not None:
                if registered_bbox_name in tracked_bboxes_names:
                    self.tracking_bboxes_names_set.add(registered_bbox_name)
                    #registered_bbox_name_item.setSelected(True)
                    self.classes_with_description_table.item(row_idx, 0).setSelected(True)

                
    def update_tracking_objects_set_from_table(self):
        rows_num = self.classes_with_description_table.rowCount()
        for row_idx in range(rows_num):
            obj_descr_item = self.classes_with_description_table.item(row_idx, 0)
            registered_bbox_name_item = self.classes_with_description_table.item(row_idx, 1)
            registered_bbox_name = registered_bbox_name_item.text()
            if obj_descr_item is not None or registered_bbox_name_item is not None:
                if obj_descr_item.isSelected() or registered_bbox_name_item.isSelected():
                    self.tracking_bboxes_names_set.add(registered_bbox_name)
                else:
                    try:
                        self.tracking_bboxes_names_set.remove(registered_bbox_name)
                    except KeyError:
                        continue
    
    def reset_table(self):
        try:
            self.classes_with_description_table.setRowCount(0)
        except:
            pass
    
    def unselect_all_table_items(self):
        rows_num = self.classes_with_description_table.rowCount()
        for row_idx in range(rows_num):
            self.classes_with_description_table.item(row_idx, 0).setSelected(False)
            self.classes_with_description_table.item(row_idx, 1).setSelected(False)



    def table_cell_click_handling(self, row, col):
        item = self.classes_with_description_table.item(row, col)
        
        # потенциально медленная операция
        self.registered_bboxes_dict, self.tracking_bboxes_dict, self.tracked_and_raw_bboxes_dict \
            = self.update_registered_and_tracking_objects_dicts('raw')
        """
        print('+++++++++TABLE_CLICK+++++++++')
        print(f'tracking_bboxes_names_set:\n{self.tracking_bboxes_names_set}')
        print(f'raw_bboxes_dict:\n{self.raw_bboxes_dict}\n')
        print(f'registered_bboxes_dict:\n{self.registered_bboxes_dict}')
        print(f'tracking_bboxes_dict:\n{self.tracking_bboxes_dict}')
        print(f'all_frames_raw_bbox_name2registered_bbox_name_dict: {self.all_frames_raw_bbox_name2registered_bbox_name_dict}')
        print('+++++++++++++++++++++++++++++')
        """

    def search_first_appearance_button_slot(self):
        self.is_autoplay = False
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
                show_info_message_box(
                    window_title="Class search info",
                    info_text="You should select only one class for searching",
                    buttons=QMessageBox.Ok,
                    icon_type=QMessageBox.Information)
                return

        if searching_class_name is None:
            show_info_message_box(
                window_title="Class search info",
                info_text="No class is selected",
                buttons=QMessageBox.Ok,
                icon_type=QMessageBox.Information)
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
                    self.read_frame(direction='forward')
                    show_info_message_box(
                        window_title="Class search info",
                        buttons=QMessageBox.Ok,
                        info_text=f"First appearance of {class_name} at frame #{frame_idx}")
                    return
        show_info_message_box(
            window_title="Class search info",
            info_text=f"{searching_class_name} is not presented", buttons=QMessageBox.Ok,
            icon_type=QMessageBox.Warning)

    def display_frame_position(self, current_frame_idx):
        if self.video_capture is None or self.frame_with_boxes is None:
            if self.imshow_thread.isRunning():
                self.stop_imshow_thread()
            return
        
        self.current_frame_display.display(current_frame_idx)
        self.current_frame_idx = current_frame_idx
        self.read_frame(direction='forward')

    def autosave_current_checkbox_slot(self):
        '''
        Обработчик checkbox, отвечающего за автоматическое сохранение кадра при переходе на новый
        '''
        self.autosave_mode = self.autosave_current_checkbox.isChecked()

    def show_tracked_checkbox_slot(self):
        self.is_tracked_bboxes_show = self.show_tracked_checkbox.isChecked()
    
    def setup_imshow_thread(self):
        '''
        При инициализации нового потока необходимо также заново подключать все сигналы
        класса потока к соответствующим слотам главного потока
        '''
        self.imshow_thread = ImshowThread()
        self.imshow_thread.bboxes_update_signal.connect(self.update_bboxes_on_frame)
        self.imshow_thread.new_bbox_create_signal.connect(self.label_new_bbox)

    def label_new_bbox(self):
        '''
        print('-----LABEL_NEW_BBOX-----')
        print('BEFORE')
        print(f'registered {self.registered_bboxes_dict}')
        print()
        print(f'tracking {self.tracking_bboxes_dict}')
        print()
        print(f'tracking {self.raw_bboxes_dict}')
        print()
        print(f'tracked_and_raw {self.tracked_and_raw_bboxes_dict}')
        print()
        '''
        self.is_autoplay = False
        label_new_bbox_dialog = LabelNewBoxDialog(
            self.frame_with_boxes.bboxes_dict['None,None'],
            self.obj_descr2registered_bbox_dict,
            self.tracked_and_raw_bboxes_dict)
        label_new_bbox_dialog.exec()

        self.frame_with_boxes.bboxes_dict.pop('None,None')
        if label_new_bbox_dialog.confirm_bbox_creation:
            created_bbox_name = label_new_bbox_dialog.current_bbox_name
            
            self.tracked_and_raw_bboxes_dict[created_bbox_name] = deepcopy(label_new_bbox_dialog.current_bbox)
            self.registered_bboxes_dict, self.tracking_bboxes_dict, self.tracked_and_raw_bboxes_dict \
                    = self.update_registered_and_tracking_objects_dicts('raw_and_tracked')
           
        '''
        print('AFTER')
        print(f'registered {self.registered_bboxes_dict}')
        print()
        print(f'tracking {self.tracking_bboxes_dict}')
        print()
        print(f'raw {self.raw_bboxes_dict}')
        print()
        print(f'tracked_and_raw {self.tracked_and_raw_bboxes_dict}')
        print('|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
        '''
    
    def update_visible_boxes_on_selection_slot(self, item):
        '''
        Обновление видимых рамок в кадре. Контролируется посредством visible_classes_list_widget.
        Если элемент выделен, то он отображается в кадре.
        '''
        return


    def load_labels_from_txt(self):
        '''
        Загружаем из txt-файлов координаты рамок и информацию о классах. 
        Информация загружается в self.frame_with_boxes, 
        self.visible_classes_list_widget не изменяется
        '''
        raise NotImplementedError

    def reset_tracker(self):
        '''
        Обнуление трекера, чтобы начать процесс детекции и трекинга заново
        '''
        self.tracker = Yolov8Tracker(model_type=self.tracker_type)



    def update_bboxes_on_frame(self):
        self.tracked_and_raw_bboxes_dict = self.frame_with_boxes.bboxes_dict
        

    def update_visible_classes_list(self):
        '''
        обновление списка рамок. 
        Информация о рамках берется из списка рамок, хранящегося в self.frame_with_boxes
        '''

        # определяем количество элементов в списке
        qlist_len = self.visible_classes_list_widget.count()

        new_list = []
        for bbox_name, bbox in self.frame_with_boxes.bboxes_dict.items():
            class_name = bbox.class_name
            sample_idx = bbox.id
            is_selected = bbox.is_visible

            displayed_name = f'{class_name},{sample_idx}'
            item = QListWidgetItem(displayed_name)
            
            for item_idx in range(qlist_len):
                prev_item = self.visible_classes_list_widget.item(item_idx)

                prev_data = prev_item.data(0)
                if prev_data == displayed_name:
                    item = prev_item
                    break

            new_list.append({'data': item.data(0), 'is_selected': is_selected})
   
        # обновление списка классов?
        self.visible_classes_list_widget.clear()
        for bbox_idx, data_dict in enumerate(new_list):
            item = QListWidgetItem(data_dict['data'])
            self.visible_classes_list_widget.addItem(item)
            self.visible_classes_list_widget.item(bbox_idx).setSelected(data_dict['is_selected'])
        
            
    def reset_display(self):
        
        #Обнуление значений на экране и на слайдере
        
        #self.frame_slider.setRange(0, 0)
        self.set_display_value(0)
        self.all_frames_display.display(0)
    
    

    def set_display_value(self, val):
        #self.frame_slider.setValue(val)
        self.current_frame_display.display(val)

    def setup_slider_range(self, max_val, current_idx):
        '''
        #Установка диапазона значений слайдера
        '''
        #self.frame_slider.setRange(0, max_val)
        self.set_slider_display_value(current_idx)
    
    def close_video(self):
        '''
        Обработчик закрытия файла
        Сохраняет рамки, закрывает поток чтения изображения и делает объект изображения с рамками пустым
        '''

        if self.frame_with_boxes is not None:
            self.save_labels()
        self.close_imshow_thread()
        self.frame_with_boxes = None
        self.reset_display()

    def read_persons_description(self):
        '''
        
        '''
        
        if not os.path.isfile(self.path_to_persons_descr):
            return 'no file'
        with open(self.path_to_persons_descr, encoding='utf-8') as fd:
            persons_descr_list = [descr for descr in fd.read().split('\n') if descr != '']
        
        if len(persons_descr_list) == 0:
            return 'empty descr'
        
        self.classes_with_description_table.setRowCount(len(persons_descr_list))
        for idx, obj_descr in enumerate(persons_descr_list):
            # заполняем общий прямой и обратный словари для взаимного отображения описаний объектов и названий зарегистрированных рамок
            self.obj_descr2registered_bbox_dict[obj_descr] = f'person,{idx}'
            self.registered_bbox2obj_descr_dict[f'person,{idx}'] = obj_descr
            
            self.classes_with_description_table.setItem(idx, 0, QTableWidgetItem(obj_descr))
            self.classes_with_description_table.setItem(idx, 1, QTableWidgetItem(f'person,{idx}'))
            
    
        
    def open_file(self):
        # закрываем поток, который отображает кадры видео
        self.close_imshow_thread()
        # обнуляем все праметры
        self.set_params_to_default()
        # обновляем трекер
        self.reset_table()
        self.reset_tracker()
        # обнуляем список классов в видео, когда загружаем новое
        #self.visible_classes_list_widget.clear()
        # получаем абсолютный путь до файла
        title = 'Open video'

        # записываем в файл settings.json путь до папки с последним открытым файлом, чтобы при следующем открытии заново не искать нужный файл
        try:
            last_opened_folder_path = self.settings_dict['last_opened_folder']
        except KeyError:
            last_opened_folder_path = '/home'

        
        # фильтр разрешений файлов
        file_filter = 'Videos (*.mp4 *.wmw *.avi *.mpeg)'
        # запускаем окно поиска файлов
        open_status_tuple = QFileDialog.getOpenFileName(self, title, last_opened_folder_path, file_filter)
        path = open_status_tuple[0]
        if len(path) == 0:
            return

        # переформатируем путь в соответствии со знаком-разделителем путей операционной системы
        path = os.sep.join(path.split('/'))
        path_to_folder, name = os.path.split(path)

        txt_name = '.'.join(name.split('.')[:-1]) + '_persons_descr.txt'
        self.path_to_persons_descr = os.path.join(path_to_folder, txt_name)
        
        #!!!!
        # это пока что не надо
        ret_status = self.read_persons_description()
        if ret_status == 'no file':
            show_info_message_box(
                window_title="Persons file description error",
                info_text=f"Person description for {name} video does not exist. Please describe persons and fill out the file {name}_persons_descr.txt",
                buttons=QMessageBox.Ok,
                icon_type=QMessageBox.Critical)
            return
        elif ret_status == 'empty descr':
            show_info_message_box(
                window_title="Persons file description error",
                info_text=f"Person description for {name} video is empty. Please describe persons and fill out the file {name}_persons_descr.txt",
                buttons=QMessageBox.Ok,
                icon_type=QMessageBox.Critical)
            return
        

        # обновляем путь до последнего открытого файла и перезаписываем файл конфигурации
        self.settings_dict['last_opened_folder'] = path_to_folder
        with open('settings.json', 'w', encoding='utf-8') as fd:
            json.dump(self.settings_dict, fd)

        # формируем путь до папки, куда будут сохраняться рамки
        label_folder_name = '.'.join(name.split('.')[:-1]) + '_labels'
        self.path_to_labelling_folder = os.path.join(path_to_folder, label_folder_name)

        if os.path.isdir(self.path_to_labelling_folder):
            # получаем список путей до txt файлов с координатами рамок номер кадра совпадает с именем файла
            self.paths_to_labels_list = glob.glob(os.path.join(self.path_to_labelling_folder, '*.txt'))
        else:
            self.paths_to_labels_list = []
            # содаем папку, куда будем сохранять рамки
            os.mkdir(self.path_to_labelling_folder)

        # создаем объект opencv для чтения видео
        self.video_capture = cv2.VideoCapture(path)
        ret, frame = self.video_capture.read()
        if not ret:
            raise RuntimeError(f'Can not read {path} video')
        
        # выясняем количество кадров
        self.frame_number = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self.all_frames_display.display(self.frame_number)

        # выясняем размер кадра
        self.img_rows, self.img_cols = frame.shape[:2]

        # выставляем счетчик кадров: если уже были сформированы рамки, то счетчик кадров делаем равным номеру последнего кадра 
        if len(self.paths_to_labels_list) > 0:
            self.current_frame_idx = len(self.paths_to_labels_list) - 1
        else:
            self.current_frame_idx = 0
        
        #self.current_frame_idx = 0

        # выставляем значение слайдера кадров
        #self.setup_slider_range(max_val=self.frame_number, current_idx=self.current_frame_idx)

        self.window_name = name
        
        # создаем объект BboxFrame, позволяющий отображать и изменять локализационные рамки на кадре видео
        self.frame_with_boxes = BboxFrameTracker(img=frame)
        
        # Подготавливаем и запускаем отдельный поток, в котором будет отображаться кадр с рамками
        # и будут изменяться рамки
        self.setup_imshow_thread()
        self.imshow_thread.setup_frame(self.frame_with_boxes, self.window_name)
        self.imshow_thread.start()

        # сразу открываем видео
        self.read_frame(direction='forward')

    def close_imshow_thread(self):
        if self.imshow_thread.isRunning():
            self.frame_with_boxes.delete_img()            
            self.imshow_thread.wait()

    def save_labels(self):
        '''
        Сохранение координат рамок и классов в json-файл, имя которого совпадает с номером кадра
        СОХРАНЕНИЕ ВЫПОЛНЯЕТСЯ АВТОМАТИЧЕСКИ ПРИ ПЕРЕХОДЕ НА СЛЕДУЮЩИЙ КАДР.
        '''
        path_to_target_json_label = os.path.join(
            self.path_to_labelling_folder, f'{self.current_frame_idx:06d}.json')
        
        # если файл с разметкой есть,то читаем его в словарь
        if os.path.isfile(path_to_target_json_label):
            with open(path_to_target_json_label, encoding='utf-8') as fd:
                labels_json_dict = json.load(fd)
        # инче создаем пустой словарь
        else:
            labels_json_dict = {}
        
        # обновляем словарь новыми рамками
        labels_json_dict.update(
            {bbox_name:[int(coord) for coord in bbox.coords] for bbox_name, bbox in self.tracking_bboxes_dict.items()})
        
        # Сохраняем разметку
        with open(path_to_target_json_label, 'w', encoding='utf-8') as fd:
            json.dump(labels_json_dict, fd, indent=4)

    def next_frame_button_handling(self):

        if self.video_capture is None or self.frame_with_boxes is None:
            self.is_autoplay = False    
            if self.imshow_thread.isRunning():
                self.stop_imshow_thread()
            return
        
        # сохраняем все рамки
        if self.current_frame_idx > -1:
            if self.autosave_mode:
                self.save_labels()
        
        # если не выбраны рамки для трекинга, то 
        if len(self.tracking_bboxes_names_set) == 0:
            self.is_autoplay = False
            show_info_message_box('Внимание!', 'Не выбраны объекты для трекинга!\nЧтобы начать воспроизведение видео выберите объекты из таблицы', QMessageBox.Ok, QMessageBox.Warning)
            return
        
        # сохраняем рамки
        self.save_labels()
        '''
        print()
        print()
        print('/////////////NEW_FRAME/////////////')
        '''
        self.current_frame_idx += 1
        if self.current_frame_idx >= self.frame_number:
            self.is_autoplay = False
            self.current_frame_idx = self.frame_number - 1
            show_info_message_box('Конец видео', 'Вы достигли конца видео', QMessageBox.Ok, QMessageBox.Information)
            return

        # сохраняем список рамок, прежде чем прочитать следующий кадр
        self.previous_tracked_and_raw_bboxes_dict = deepcopy(self.tracked_and_raw_bboxes_dict)

        self.read_frame(direction='forward')

        # при переходе на новый кадр обнуляем словарь для отображения  
        # имен сгенерированных рамок на имена зарегистирированных рамок
        self.current_frame_raw_bbox_name2registered_bbox_name_mapping = One2OneMapping()
        return

    def previous_frame_button_handling(self):
        if self.video_capture is None or self.frame_with_boxes is None:
            self.is_autoplay = False
            if self.imshow_thread.isRunning():
                self.stop_imshow_thread()
            return

        # сохраняем рамки
        self.save_labels()

        self.current_frame_idx -= 1
        if self.current_frame_idx < 0:
            self.is_autoplay = False
            return
               
        # сохраняем список рамок, прежде чем прочитать следующий кадр
        self.previous_tracked_and_raw_bboxes_dict = deepcopy(self.tracked_and_raw_bboxes_dict)

        self.read_frame(direction='backward')

        # при переходе на новый кадр обнуляем словарь для отображения  
        # имен сгенерированных рамок на имена зарегистирированных рамок
        self.current_frame_raw_bbox_name2registered_bbox_name_mapping = One2OneMapping()

    def autoplay(self):
        self.is_autoplay = True
        for i in range(30):
            if not self.is_autoplay:
                break
            self.next_frame_button_handling()

        
          
    def try_alternative_tracking(self, bbox_name):
        automatically_tracked_bbox = {}
        prev_bbox = self.previous_tracked_and_raw_bboxes_dict[bbox_name]
        
        if prev_bbox.is_additionaly_tracked:
            # читаем предыдущий кадр
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx-1)
            _, prev_frame = self.video_capture.read()
            
            # получаем рамку на предыдущем кадре
            
            prev_coords = prev_bbox.x0y0wh()
            try:
                # объявляем трекер
                alternative_tracker = cv2.legacy.TrackerMOSSE_create()
                alternative_tracker.init(prev_frame, prev_coords)
                _, tracked_bbox_coords = alternative_tracker.update(self.frame_with_boxes.img)
            except:
                return automatically_tracked_bbox
            
            tracked_bbox_coords = xywh2xyxy(*tracked_bbox_coords)
            new_coords = process_box_coords(*tracked_bbox_coords, self.img_rows, self.img_cols)
            new_area = compute_bbox_area(*new_coords)
            if new_area < 16:
                # если площадь рамки стала слишком маенькой, надо предупредить об этом
                # если с рамкой произошла какая-то беда, то оставляем предыдущие координаты
                new_coords = xywh2xyxy(*prev_coords)
            #self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            new_bbox = deepcopy(prev_bbox)
            new_bbox.color = prev_bbox.color
            new_bbox.coords = new_coords
            new_bbox.is_additionaly_tracked = True
            automatically_tracked_bbox = {bbox_name: new_bbox}

        return automatically_tracked_bbox

        
    def compare_tracked_and_raw_bboxes_dicts(self):
        current_bboxes_names_set = set(self.tracked_and_raw_bboxes_dict.keys())
        prev_bboxes_names_set = set(self.previous_tracked_and_raw_bboxes_dict.keys())
        new_bboxes_names = current_bboxes_names_set - prev_bboxes_names_set
        disappeared_bboxes_names = prev_bboxes_names_set - current_bboxes_names_set
        '''
        print('----COMPARE_TRACKED_AND_RAW----')
        print(f'current_bboxes_names_set: {current_bboxes_names_set}')
        print(f'prev_bboxes_names_set: {prev_bboxes_names_set}')
        print()
        print(f'new_bboxes_names: {new_bboxes_names}')
        print(f'disappeared_bboxes_names: {disappeared_bboxes_names}')
        print('-------------------------------')
        '''
        # смотрим, какие отслеживаемые рамки исчезли.
        # Нас не очень интересует, какие не отслеживаемые рамки исчезли
        for disappeared_bbox_name in disappeared_bboxes_names:
            # (R) - это признак автоматически сгенерированной рамки
            if '(R)' not in disappeared_bbox_name:
                # сначала пытаемся выполнить "альтерантивный" трекинг
                new_bbox_dict = self.try_alternative_tracking(disappeared_bbox_name)
                if len(new_bbox_dict) != 0:
                    self.tracked_and_raw_bboxes_dict.update(new_bbox_dict)
                    self.registered_bboxes_dict, self.tracking_bboxes_dict, self.tracked_and_raw_bboxes_dict \
                        = self.update_registered_and_tracking_objects_dicts('raw_and_tracked')
                    continue
                
                # пытаемся вручную ассоциировать пропавший объект с самой ближней вновь сгенерированными рамкой
                disappeared_bbox = self.previous_tracked_and_raw_bboxes_dict[disappeared_bbox_name]
                min_dist = 10e18
                nearest_name = None
                # ищем минимальное Евклидово расстояние от пропавшей рамки до вновь сгенерированной
                for raw_name, bbox in self.raw_bboxes_dict.items():
                    dist = np.linalg.norm(bbox.numpy_coords()-disappeared_bbox.numpy_coords())
                    if dist < min_dist:
                        min_dist = dist
                        nearest_name = raw_name

                if nearest_name is not None:
                    try:
                        # проверяем, не является ли ближайшей рамкой та, которая уже отслеживается
                        potential_tracking_bbox_name = self.all_frames_raw_bbox_name2registered_bbox_name_dict[nearest_name]
                        if potential_tracking_bbox_name not in self.tracking_bboxes_dict.keys():
                            raise Exception
                    except:
                        result = show_info_message_box(
                            'Отслеживаемый объект пропал',
                            f'Пропал объект {disappeared_bbox_name}. Наиболее похожая автоматически сгенерированная рамка: {nearest_name}. Выполнить ее регистрацию?',
                            QMessageBox.Yes|QMessageBox.No,
                            QMessageBox.Warning
                        )
                        self.is_autoplay = False
                        if result == QMessageBox.Yes:
                            ret = self.register_persons_handling()
                            if ret is None:
                                continue

                # потом пытаемся выполнить альтернативный трекинг
                result = show_info_message_box(
                        'Отслеживаемый объект пропал',
                        f'Пропал объект {disappeared_bbox_name}. Выполнить дополнительную попытку трекинга?',
                        QMessageBox.Yes|QMessageBox.No,
                        QMessageBox.Warning
                    )
                self.is_autoplay = False
                if result == QMessageBox.Yes:                    
                    # заставляем рамку на предыдущем шаге быть восприимчивой к доп. трекингу
                    self.previous_tracked_and_raw_bboxes_dict[disappeared_bbox_name].is_additionaly_tracked = True
                    new_bbox_dict = self.try_alternative_tracking(disappeared_bbox_name)
                    self.tracked_and_raw_bboxes_dict.update(new_bbox_dict)
                    self.registered_bboxes_dict, self.tracking_bboxes_dict, self.tracked_and_raw_bboxes_dict \
                        = self.update_registered_and_tracking_objects_dicts('raw_and_tracked')

        # тут мы смотрим, какие рамки появились
        if len(new_bboxes_names) != 0:
            if self.get_tracked_bboxes_num() == 0 or self.check_bboxes_additionaly_tracked():
                new_bboxes_names_str = '\n'.join(list(new_bboxes_names))
                msg_str = f'Появились объекты:\n{new_bboxes_names_str}\nВыполнить дополнительную регистрацию?'
                result = show_info_message_box(
                        'Появились новые объекты',
                        msg_str,
                        QMessageBox.Yes|QMessageBox.No,
                        QMessageBox.Warning
                    )
                self.is_autoplay = False
                if result == QMessageBox.Yes:
                    self.register_persons_handling()
    

    def check_bboxes_additionaly_tracked(self):
        is_additionaly_tracked = False
        for bbox_name, bbox in self.tracked_and_raw_bboxes_dict.items():
            if bbox.is_additionaly_tracked:
                return True
        return is_additionaly_tracked

    def get_tracked_bboxes_num(self):
        '''
        Получение количества отслеживаемых рамок нужно для
        - контроля появления новых объектов в кадре
        '''
        tracked_num = 0
        for bbox_name, bbox in self.tracked_and_raw_bboxes_dict.items():
            if bbox.color == (0, 255, 0):
                tracked_num += 1
        return tracked_num

    def read_frame(self, direction):
        # проверка условий возможности чтения кадра: отсутствие объекта, отвечающего за чтение кадров видео
        # или номер текущего кадра превышает количество кадров в видео
        if self.video_capture is None or self.current_frame_idx >= self.frame_number:
            return
        
        if self.current_frame_idx < 0:
            self.current_frame_idx = 0
            return
        
        # устанавливаем положение слайдера
        self.set_display_value(self.current_frame_idx)
        
        # выставляем в объекте чтения кадров позицию текущего кадра, чтобы иметь возможность двигаться не только вперед, но и назад
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        # читаем текущий кадр
        ret, frame = self.video_capture.read()
        '''
        if self.img_cols / self.screen_width > 0.65 or self.img_rows / self.screen_height > 0.65:
            scaling_factor = 0.65*self.screen_width/self.img_cols
            new_size = tuple(map(lambda x: int(scaling_factor*x), (self.img_cols, self.img_rows)))
            frame = cv2.resize(frame, new_size)
        '''

        if ret:
            # обновляем отображаемые на видео рамки
            self.frame_with_boxes.update_img(frame)
            
            # выполняем трекинг
            try:
                self.raw_bboxes_dict = self.tracker.track(
                    target_class_name='person',
                    source=frame,
                    persist=True,
                    verbose=True
                    )
            except:
                pass
                                  
            self.registered_bboxes_dict, self.tracking_bboxes_dict, self.tracked_and_raw_bboxes_dict \
                = self.update_registered_and_tracking_objects_dicts('raw')
            
            self.frame_with_boxes.bboxes_dict = self.tracked_and_raw_bboxes_dict


            # сравниваем рамки текущего кадра с рамками предыдущего кадра
            self.compare_tracked_and_raw_bboxes_dicts()

            self.frame_with_boxes.bboxes_dict = self.tracked_and_raw_bboxes_dict
            
            '''
            print('-------------READ_FRAME-------------')
            print(f'tracking_bboxes_names_set:\n{self.tracking_bboxes_names_set}')
            print(f'raw_bboxes_dict:\n{self.raw_bboxes_dict}\n')
            print(f'registered_bboxes_dict:\n{self.registered_bboxes_dict}')
            print(f'tracking_bboxes_dict:\n{self.tracking_bboxes_dict}')
            print(f'all_frames_raw_bbox_name2registered_bbox_name_dict: {self.all_frames_raw_bbox_name2registered_bbox_name_dict}')
            print('-------------------------------')
            '''     
            
    def stop_showing(self):
        if self.is_showing:
            self.is_showing = False
            cv2.destroyAllWindows()

class Yolov8Tracker:
    def __init__(self, model_type='yolov8n.pt'):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # debug!
        #device = torch.device('cpu')
        
        print(f'We are using {device} device for tracking')
        print()
        self.tracker = YOLO(model_type).to(device)
        # словарь для перевода имени класса в номер метки
        self.name2class_idx = {val: key for key, val in self.tracker.names.items()}
        
        # словарь индексов, кторые получаются в результате трекинга
        self.raw_tracking_indices_dict = {}
        # итоговый словарь индексов объектов
        self.resulting_indices_dict = {}

    def track(self, target_class_name, *yolo_args, **yolo_kwargs):
        '''
        target_class_name - имя класса, который мы собираемся детектировать
        '''
        # выполнение трекинга 
        results = self.tracker.track(*yolo_args, **yolo_kwargs)[0]
        # получение координат рамок
        bboxes = results.boxes.xyxy.long().numpy()
        # получение индексов объектов
        ids = results.boxes.id.long().numpy()
        
        # получение списка детектированных классов
        detected_classes = results.boxes.cls.long().numpy()
        # формирование индексов, фильтрующих нужный нам класс детектируемых объектов
        target_classes_filter = detected_classes == self.name2class_idx[target_class_name]

        # Фильтрация индексов объектов, рамок
        ids = ids[target_classes_filter]
        bboxes = bboxes[target_classes_filter]

        # строки и столбцы изображения нужны для создания объектов рамок
        img_rows, img_cols = results[0].orig_img.shape[:-1]
        
        # этот параметр нужен, чтобы рамка строилась не впритык объекту, а захватывала еще некоторую дополнительную область
        bbox_append_value = int(min(img_rows, img_cols)*0.025)

        bboxes_dict = {}
        for bbox, id in zip(bboxes, ids):
            #class_name = f'{target_class_name}{id:03d}'
            class_name = f'{target_class_name}(R)'
            x0,y0,x1,y1 = bbox
            # добавляем несколько пикселей, чтобы рамка строилась не впритык
            x0,y0,x1,y1 = x0 - bbox_append_value, y0 - bbox_append_value, x1 + bbox_append_value, y1 + bbox_append_value
            x0,y0,x1,y1 = process_box_coords(x0,y0,x1,y1, img_rows, img_cols)
            # пока что оставляем всего лишь один цвет - 
            color = (0,0,0)
            is_visible=True
            bboxes_dict[f'{class_name},{id}'] = Bbox(x0, y0, x1, y1, img_rows, img_cols, class_name, color, id, is_visible)

        return bboxes_dict
    
    #def register_bboxes


class ImshowThread(QThread):
    bboxes_update_signal = pyqtSignal()
    new_bbox_create_signal = pyqtSignal()
    

    def  __init__(self, parent=None):
        super().__init__(parent)
        self.frame_with_boxes = None
        self.window_name = None
        self.is_showing = False
        self.is_drawing = False
    
    def setup_frame(self, frame_with_boxes, window_name):
        self.frame_with_boxes = frame_with_boxes
        self.window_name = window_name

    def run(self):
        self.init_showing_window()
        # почему-то работает только это условие...
        # потом надо переписать,
        while self.frame_with_boxes.img is not None:
            if self.frame_with_boxes.is_bboxes_changed:
                self.bboxes_update_signal.emit()
                self.frame_with_boxes.is_bboxes_changed = False

            if self.frame_with_boxes.is_bboxes_dragged:
                self.bboxes_update_signal.emit()
                self.frame_with_boxes.is_bboxes_dragged = False

            if self.frame_with_boxes.is_bbox_created:
                self.new_bbox_create_signal.emit()
                self.frame_with_boxes.is_bbox_created = False


            img_with_boxes = self.frame_with_boxes.render_boxes()
            cv2.imshow(self.window_name, img_with_boxes)
            
            key = cv2.waitKey(20)

        self.frame_with_boxes = None
        self.stop_showing()

    def init_showing_window(self):
        if not self.is_showing:
            self.is_showing = True
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self.frame_with_boxes)

    def stop_showing(self):
        
        if self.is_showing:
            self.is_showing = False
            cv2.destroyAllWindows()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    screen_resolution = app.desktop().screenGeometry()
    ex = TrackerWindow(screen_resolution.width(), screen_resolution.height())
    
    sys.exit(app.exec_())