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

import pandas as pd

from copy import deepcopy

import numpy as np

import shutil

import time
import torch

from new_opencv_frames import BboxFrameTracker, Bbox, BboxesContainer, process_box_coords, xywh2xyxy, compute_bbox_area

from ultralytics import YOLO

def show_info_message_box(window_title, info_text, buttons, icon_type):
    '''
    Обертка для вызова диалоговых окон
    '''
    msg_box = QMessageBox()
    msg_box.setIcon(icon_type)
    msg_box.setWindowTitle(window_title)
    msg_box.setText(info_text)
    msg_box.setStandardButtons(buttons)
    return msg_box.exec()

class One2OneMapping:
    '''
    Реализация однозначных отображений. Нужен для работы рамок
    '''
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
    def __init__(self, registered_objects_container):
        '''
        Данное диалоговое окно служит для выбора человека, движение которого мы отслеживаем
        Диалоговое окно должно выскакивать после первого кадра.
        '''
        super().__init__()

        self.registered_objects_container = registered_objects_container
        
        self.setMinimumWidth(250)

        # показатели, которые мы меняем
        self.current_class_name = '---'
        self.current_object_descr = ''
        self.current_registered_idx = -1

        self.setWindowTitle('Присваивание имени новой рамке')

        self.confirm_button = QPushButton('Сохранить')
        self.cancell_button = QPushButton('Отменить')
        self.registered_bboxes_combobox = QComboBox()
        
        # заполнение выпадающих списков значениями
        registered_bboxes = registered_objects_container.get_registered_objects_db()
        registered_bboxes_list = [
            f"{row['object_description']},{row['class_name']},{row['object_idx']}" for idx, row in registered_bboxes.iterrows()]
        self.registered_bboxes_combobox.addItems(['---']+registered_bboxes_list)

        self.registered_bboxes_combobox.activated[str].connect(self.combobox_value_changed)
        self.confirm_button.clicked.connect(self.confirm_and_exit)
        self.cancell_button.clicked.connect(self.cancell_and_exit)

        functional_layout = QHBoxLayout()
        functional_layout.addWidget(self.registered_bboxes_combobox)

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
        if self.current_class_name == '---':
            self.cancell_and_exit()
        self.close()

    def cancell_and_exit(self):
        self.current_class_name == '---'
        self.current_object_descr = ''
        self.current_registered_idx = -1
        self.close()

    def combobox_value_changed(self, value):
        if value == '---':
            self.current_class_name == '---'
            self.current_object_descr = ''
            self.current_registered_idx = -1
            return
        
        self.current_object_descr, self.current_class_name, self.current_registered_idx = value.split(',')
        self.current_registered_idx = int(self.current_registered_idx)

class RegisterTrackingObjectsDialog(QDialog):
    def __init__(self, registered_objects_container, available_classes_list):
        '''
        Данное диалоговое окно служит для выбора человека, движение которого мы отслеживаем
        Диалоговое окно должно выскакивать после первого кадра.
        registered_objects_container - контейнер (), где хранятся все классы отслеживаемых объектов с индексами
        available_classes_list - список доступных классов 
        registration_type - тип регистрации - полная, выбор класса, введение информации
        '''
        super().__init__()
        
        self.setMinimumWidth(300)

        # контейнер, хранящий имена зарегистрированных объектов и их индексы
        self.registered_objects_container = registered_objects_container

        self.setWindowTitle('Регистрация отслежиываемых объектов')

        # названия колонок
        registered_bboxes_label = QLabel(text='Выбор из\nзарегистрированных\nрамок')
        register_new_object_label = QLabel(text='Регистрация нового объекта')
        assign_auto_bbox_label = QLabel(text='Присвоение автоматически\nсгенерированной рамки')

        # коклонка регистрации
        print_obj_descr_label = QLabel(text='Введите описание:')
        choose_obj_class = QLabel(text='Выберите класс:')

        # поле для ввода описания класса
        self.object_descr_textline = QLineEdit()
        self.object_descr_textline.setText('')
        
        self.save_and_exit_button = QPushButton('Сохранить и выйти')
        self.exit_without_save_button = QPushButton('Выйти без сохранения')
        # Выпадающий список с доступными классами обънектов
        self.available_class_names_combobox = QComboBox()
        # выпадающий список с зарегистрированными объектами
        self.registered_bboxes_combobox = QComboBox()
        # выпадающий список с автоматически сгенерированными рамками
        self.autogenerated_bboxes_combobox = QComboBox()
        
        # заполнение выпадающих списков значениями
        #
        # доступные имена классов
        available_classes_list = sorted(available_classes_list)
        self.available_class_names_combobox.addItems(['---']+available_classes_list)
        # зарегистрированные рамки
        registered_bboxes = registered_objects_container.get_registered_objects_db()
        registered_bboxes_list = [
            f"{row['object_description']} {row['class_name']},{row['object_idx']}" for idx, row in registered_bboxes.iterrows()]
        self.registered_bboxes_combobox.addItems(['---']+registered_bboxes_list)
        # автоматически сгенерированные рамки
        autogenerated_bboxes = registered_objects_container.get_all_autogenerated_bboxes()
        autogenerated_bboxes_list = [
            f"{row['object_description']} {row['bbox'].class_name},(AG){row['bbox'].auto_idx}" for idx, row in autogenerated_bboxes.iterrows()]
        self.autogenerated_bboxes_combobox.addItems(['---']+autogenerated_bboxes_list)

        self.object_descr_textline.textChanged[str].connect(self.update_object_desr)
        self.available_class_names_combobox.activated[str].connect(self.available_class_names_combobox_value_changed)
        self.registered_bboxes_combobox.activated[str].connect(self.bboxes_combobox_value_changed)

        self.save_and_exit_button.clicked.connect(self.save_and_exit)
        self.exit_without_save_button.clicked.connect(self.exit_without_save)
        
        # Присвоение значений списков по умолчанию
        self.current_class_name = '---'
        self.current_object_descr = ''
        #self.current_auto_bbox = None

        # левая колонка
        registered_layout = QVBoxLayout()
        registered_layout.addWidget(registered_bboxes_label)
        registered_layout.addWidget(self.registered_bboxes_combobox)

        # средняя колонка
        register_new_labels_layout = QVBoxLayout()
        register_new_labels_layout.addWidget(print_obj_descr_label)
        register_new_labels_layout.addWidget(choose_obj_class)

        register_new_funct_layout = QVBoxLayout()
        register_new_funct_layout.addWidget(self.object_descr_textline)
        register_new_funct_layout.addWidget(self.available_class_names_combobox)

        register_new_layout = QHBoxLayout()
        register_new_layout.addLayout(register_new_labels_layout)        
        register_new_layout.addLayout(register_new_funct_layout)

        # Правая колонка
        assign_existent_layout = QVBoxLayout()
        assign_existent_layout.addWidget(assign_auto_bbox_label)
        assign_existent_layout.addWidget(self.autogenerated_bboxes_combobox)

        functional_layout = QHBoxLayout()
        functional_layout.addLayout(register_new_layout)
        
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.save_and_exit_button)
        buttons_layout.addWidget(self.exit_without_save_button)
                
        main_layout = QVBoxLayout()
        main_layout.addLayout(functional_layout)
        main_layout.addLayout(buttons_layout)
        self.setLayout(main_layout)

    def update_object_desr(self, text):
        self.current_object_descr = text

    def exit_without_save(self):
        self.current_class_name == '---'
        self.current_object_descr = ''
        self.close()

    def save_and_exit(self):
        '''
        Меняем имя класса и номер у рамки
        '''
        if self.current_class_name == '---':
            show_info_message_box('Нет изменений', 'Не выбрано имя класса', QMessageBox.Ok, QMessageBox.Critical)
            return
                
        self.close()

    def bboxes_combobox_value_changed(self, value):
        # ищем рамку с теми же самыми именем класса и индексом
        #!!!!
        self.current_raw_bbox_name = value

    def available_class_names_combobox_value_changed(self, value):
        self.current_class_name = value

class DeleteTrackingObjectsDialog(QDialog):
    def __init__(self, registered_objects_container):
        '''
        Данное диалоговое окно служит для выбора человека, движение которого мы отслеживаем
        Диалоговое окно должно выскакивать после первого кадра.
        registered_objects_container - контейнер (), где хранятся все классы отслеживаемых объектов с индексами
        available_classes_list - список доступных классов 
        registration_type - тип регистрации - полная, выбор класса, введение информации
        '''
        super().__init__()
        
        self.setMinimumWidth(300)

        # контейнер, хранящий имена зарегистрированных объектов и их индексы
        self.registered_objects_container = registered_objects_container

        self.setWindowTitle('Удаление отслежиываемых объектов')
        
        choose_removing_obj_label = QLabel(text='Выберите объект из списка:')
        
        self.save_and_exit_button = QPushButton('Сохранить и выйти')
        self.exit_without_save_button = QPushButton('Выйти без сохранения')

        # выпадающий список с зарегистрированными объектами
        self.registered_bboxes_combobox = QComboBox()

        # заполнение выпадающих списков значениями
        #
        # зарегистрированные рамки
        registered_bboxes = registered_objects_container.get_registered_objects_db()
        registered_bboxes_list = [
            f"{row['object_description']},{row['class_name']},{row['object_idx']}" for idx, row in registered_bboxes.iterrows()]
        self.registered_bboxes_combobox.addItems(['---']+registered_bboxes_list)
        
        self.registered_bboxes_combobox.activated[str].connect(self.bboxes_combobox_value_changed)

        self.save_and_exit_button.clicked.connect(self.save_and_exit)
        self.exit_without_save_button.clicked.connect(self.exit_without_save)
        
        # Присвоение значений списков по умолчанию
        self.current_class_name = '---'
        self.current_object_descr = ''
        self.current_class_idx = -1
        
        functional_layout = QHBoxLayout()
        
        functional_layout.addWidget(choose_removing_obj_label)
        functional_layout.addWidget(self.registered_bboxes_combobox)
        
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.save_and_exit_button)
        buttons_layout.addWidget(self.exit_without_save_button)
                
        main_layout = QVBoxLayout()
        main_layout.addLayout(functional_layout)
        main_layout.addLayout(buttons_layout)
        self.setLayout(main_layout)

    def update_object_desr(self, text):
        self.current_object_descr = text

    def exit_without_save(self):
        self.current_class_name == '---'
        self.current_class_idx = -1
        self.current_object_descr = ''
        self.close()

    def save_and_exit(self):
        '''
        Меняем имя класса и номер у рамки
        '''
        if self.current_class_name == '---':
            show_info_message_box('Нет изменений', 'Не выбрано имя класса', QMessageBox.Ok, QMessageBox.Critical)
            return
        
        result = show_info_message_box(
            'УДАЛЕНИЕ ОБЪЕКТА!',
            'ВНИМАНИЕ!\nВЫ УДАЛЯЕТЕ ОБЪЕКТ ИЗ БАЗЫ\nОТСЛЕЖИВАЕМЫХ ОБЪЕКТОВ\nЭТУ ОПЕРАЦИЮ НЕЛЬЗЯ ОТМЕНИТЬ\nПРОДОЛЖИТЬ?',
            buttons=QMessageBox.Yes | QMessageBox.No,
                icon_type=QMessageBox.Warning
            )
        if result == QMessageBox.No:
            return
        
        self.close()

    def bboxes_combobox_value_changed(self, value):
        # ищем рамку с теми же самыми именем класса и индексом
        description, class_name, class_idx = value.split(',')
        self.current_object_descr = description
        self.current_class_name = class_name
        self.current_class_idx = int(class_idx)

class AssociateRegisteredAndAutoBboxesDialog(QDialog):
    def __init__(self, registered_objects_container):
        '''
        Данное диалоговое окно служит для выбора человека, движение которого мы отслеживаем
        Диалоговое окно должно выскакивать после первого кадра.
        '''
        super().__init__()
        
        self.setMinimumWidth(250)

        # показвтели, которые мы меняем
        self.current_class_name = '---'
        self.current_auto_idx = -1
        self.current_registered_class_name = '---'
        self.current_registered_idx = -1
        self.current_registered_object_descr = '---'
        
        # контейнер, хранящий имена зарегистрированных объектов и их индексы
        self.registered_objects_container = registered_objects_container

        self.setWindowTitle('Ассоциация автоматических и зарегистрированных рамок')
        
        self.save_and_exit_button = QPushButton('Сохранить и выйти')
        self.exit_without_save_button = QPushButton('Выйти без сохранения')
        self.registered_objects_combobox = QComboBox()
        self.auto_bboxes_combobox = QComboBox()
        
        # заполнение выпадающих списков значениями
        registered_bboxes = registered_objects_container.get_registered_objects_db()
        registered_bboxes_list = [
            f"{row['object_description']},{row['class_name']},{row['object_idx']}" for idx, row in registered_bboxes.iterrows()]
        self.registered_objects_combobox.addItems(['---']+registered_bboxes_list)
        
        auto_bboxes = registered_objects_container.get_all_autogenerated_bboxes()
        auto_bboxes_list = [
            f"{row['class_name']}(AG),{row['auto_idx']}" for idx, row in auto_bboxes.iterrows()]
        self.auto_bboxes_combobox.addItems(['---']+auto_bboxes_list)
        
        self.registered_objects_combobox.activated[str].connect(self.registered_objects_combobox_value_changed)
        self.auto_bboxes_combobox.activated[str].connect(self.auto_bboxes_combobox_value_changed)

        self.save_and_exit_button.clicked.connect(self.save_and_exit)
        self.exit_without_save_button.clicked.connect(self.exit_without_save)

        titles_layout = QVBoxLayout()
        titles_layout.addWidget(QLabel(text='Автоматически сгенерированная рамка:'))
        titles_layout.addWidget(QLabel(text='Зарегистрированный объект:'))
        comboboxes_layout = QVBoxLayout()
        comboboxes_layout.addWidget(self.auto_bboxes_combobox)
        comboboxes_layout.addWidget(self.registered_objects_combobox)
        functional_layout = QHBoxLayout()
        functional_layout.addLayout(titles_layout)
        functional_layout.addLayout(comboboxes_layout)
        

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.save_and_exit_button)
        buttons_layout.addWidget(self.exit_without_save_button)
                

        main_layout = QVBoxLayout()
        main_layout.addLayout(functional_layout)
        main_layout.addLayout(buttons_layout)
        self.setLayout(main_layout)

    def exit_without_save(self):
        self.current_class_name = '---'
        self.current_auto_idx = -1
        self.current_registered_class_name = '---'
        self.current_registered_idx = -1
        self.current_registered_object_descr = '---'
        self.close()

    def save_and_exit(self):
        '''
        Меняем имя класса и номер у рамки
        '''
        if self.current_registered_class_name == '---':
            show_info_message_box('Нет изменений', 'Не выбрана автоматически сгенерированная рамка', QMessageBox.Ok, QMessageBox.Critical)
            return
        if self.current_class_name == '---':
            show_info_message_box('Нет изменений', 'Не выбран зарегистрированный объект', QMessageBox.Ok, QMessageBox.Critical)
            return
        
        self.close()

    def registered_objects_combobox_value_changed(self, value):
        if value == '---':
            self.current_registered_object_descr = '---'
            self.current_registered_class_name = '---'
            self.current_registered_idx = -1
        else:
            self.current_registered_object_descr, self.current_registered_class_name, self.current_registered_idx = value.split(',')
            self.current_registered_idx = int(self.current_registered_idx)

    def auto_bboxes_combobox_value_changed(self, value):
        if value == '---':
            self.current_class_name = '---'
            self.current_auto_idx = -1
        else:
            self.current_class_name, self.current_auto_idx  = value.split('(AG),')
            self.current_auto_idx = int(self.current_auto_idx)


class SetFrameIdxDialog(QDialog):
    def __init__(self, frames_num):
        '''
        Диалоговое окно для перехода на заданный кадр видео
        '''
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


class SelectModelDialog(QDialog):
    def __init__(self, models_names_list):
        '''
        Диалоговое окно для выбора модели детектора/доп. трекера
        '''
        super().__init__()
        
        self.combobox = QComboBox()
        self.combobox.addItems(models_names_list)
        self.combobox.activated[str].connect(self.select_detector)
        self.current_model = models_names_list[0]#'yolov8x' if torch.cuda.is_available() else 'yolov8s'

        self.buttons = QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.combobox)
        self.main_layout.addWidget(self.buttons)
        self.setLayout(self.main_layout)

    def select_detector(self, val):
        self.current_model = val

class AddTrackingObjectDialog(QDialog):
    def __init__(self, available_classes_list):
        super().__init__()

        self.selected_class_name = None
        self.object_descr = ''

        available_classes_list = sorted(available_classes_list)
        self.available_classes_list = available_classes_list
        self.available_classes_combobox = QCheckBox()
        self.available_classes_combobox.addItems(self.available_classes_list)
        self.available_classes_combobox.activated[str].connect(self.select_object_class)

        self.object_description_textline = QLineEdit()
        self.obj_descr_textline.textChanged[str].connect(self.write_obj_descr)

        self.save_and_exit_button = QPushButton('Сохранить и выйти')
        self.exit_without_save_button = QPushButton('Выйти без сохранения')

    def select_object_class(self, class_name):
        self.selected_class_name = class_name

    def write_obj_descr(self, text):
        self.object_descr = text

    def save_and_exit(self):
        if self.selected_class_name is None:
            show_info_message_box(
                'ВНИМАНИЕ!',
                'Имя класса отслеживаемого объекта не выбрано!',
                QMessageBox.Ok,
                QMessageBox.Warning
            )
            return
        self.close()

    def exit_without_save(self):
        self.selected_class_name = None
        self.object_descr = ''
        self.close()

class TrackerWindow(QMainWindow):
    def __init__(self, screen_width, screen_height, tracker_type='yolov8x.pt'):
        '''
        Главный класс приложения
        '''
        super().__init__()
        # DEBUG
        #shutil.rmtree('debug_bbox_update', ignore_errors=True)
        #os.makedirs('debug_bbox_update', exist_ok=True)

        # т.к. обнулять параметры надо в нескольких местах, эта процедура обернуты в один метод
        self.set_all_params_to_default()

        self.detectors_names_list = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']      
        self.tracker_type = self.detectors_names_list[-1] if torch.cuda.is_available() else self.detectors_names_list[-4]
        self.tracker = YoloTracker(model_type=self.tracker_type)

        self.additional_trackers_create_functions_dict = {
            'BOOSTING':cv2.legacy.TrackerBoosting_create,
            'MIL':cv2.TrackerMIL_create,
            'MEDIANFLOW':cv2.legacy.TrackerMedianFlow_create,
            'KCF':cv2.TrackerKCF_create,
            'MOSSE':cv2.legacy.TrackerMOSSE_create,
            'CSRT':cv2.TrackerCSRT_create}
        self.additional_tracker_type = 'CSRT'
        # отдельный дополнительный трекер для каждой рамки
        self.additional_trackers_dict = {} # self.additional_trackers_create_functions_dict[self.additional_tracker_type]()

        self.screen_width = screen_width
        self.screen_height = screen_height

        self.is_logging_checkbox = QCheckBox('ВКЛЮЧИТЬ ЗАПИСЬ ЛОГА')
        self.is_logging_checkbox.setChecked(True)

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
        #autoplay_button = QPushButton("Autoplay 30 frames")
        self.disable_add_tracking = QCheckBox('Откл. доп трекинг')

        register_objects_button = QPushButton('Регистрация нового объекта')
        delete_registered_objects_button = QPushButton('Удаление зарегистрированного объекта')
        associate_auto_bboxes_button = QPushButton('Ассоциировать автоматические рамки')
        cancell_register_objects_button = QPushButton('Отмена регистрации всех объектов')
        
        reset_tracker_and_set_frame_button = QPushButton('Сбросить трекер и перейти на заданный кадр')
        
        show_auto_bboxes_button = QPushButton('Показать только автоматически сгенерированные рамки')
        show_tracked_button = QPushButton('Показать только отслеживаемые рамки')
        show_tracked_and_raw_button = QPushButton('Показать отслеживаемые и сгенерированные рамки')
        

        self.classes_with_description_table = QTableWidget()
        self.classes_with_description_table.setColumnCount(4)
        self.classes_with_description_table.setHorizontalHeaderLabels(["№", "Класс", "Описание объекта", "Автоматическая рамка"])
        self.classes_with_description_table.setColumnWidth(0, 30)
        self.classes_with_description_table.setColumnWidth(1, 70)
        self.classes_with_description_table.setColumnWidth(2, 170)
        self.classes_with_description_table.setColumnWidth(2, 170)
        self.classes_with_description_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # чтение списка классов из json
        with open('settings.json', 'r', encoding='utf-8') as fd:
            self.settings_dict = json.load(fd)

        self.current_detector_label = QLabel(text=f'Текущий НС детектор: {self.tracker_type.split(".")[0]}')
        self.current_additional_tracker_label = QLabel(text=f'Текущий доп. трэкер: {self.additional_tracker_type}')

        self.current_frame_label = QLabel()
        self.current_frame_label.setText('Идекс текущего кадра:')
        self.current_frame_display = QLCDNumber()
        self.frames_num_label = QLabel()
        self.frames_num_label.setText('Общее количество кадров:')
        self.all_frames_display = QLCDNumber()
        self.reset_display()

        # присоединение к обработчику события
        next_frame_button.clicked.connect(self.next_frame_button_handling)
        previous_frame_button.clicked.connect(self.previous_frame_button_handling)
        #autoplay_button.clicked.connect(self.autoplay)
        delete_registered_objects_button.clicked.connect(self.delete_registered_handling)
        register_objects_button.clicked.connect(self.register_objects_handling)
        associate_auto_bboxes_button.clicked.connect(self.associate_auto_bboxes_handling)
        cancell_register_objects_button.clicked.connect(self.cancell_register_objects_button_handling)
        reset_tracker_and_set_frame_button.clicked.connect(self.reset_tracker_and_set_frame_button_handling)
        show_auto_bboxes_button.clicked.connect(self.show_auto_bboxes_button_handling)
        show_tracked_button.clicked.connect(self.show_registered_bboxes_button_handling)
        show_tracked_and_raw_button.clicked.connect(self.show_regstered_and_auto_bboxes_button_handling)
        self.disable_add_tracking.stateChanged.connect(self.disable_add_tracking_slot)

        self.classes_with_description_table.cellClicked.connect(self.table_cell_click_handling)
        self.classes_with_description_table.cellEntered.connect(self.table_cell_click_handling)

        # действия для строки меню
        open_file = QAction('Open', self)
        open_file.setShortcut('Ctrl+O')
        open_file.triggered.connect(self.open_file)
        close_file = QAction('Close', self)
        close_file.triggered.connect(self.close_video)


        change_detector = QAction('Change Detector Type', self)
        change_additional_tracker = QAction('Change additional tracker', self)
        change_detector.triggered.connect(self.change_detector_handling)
        change_additional_tracker.triggered.connect(self.change_additional_tracker_handling)
       
        # строка меню
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        editMenu = menubar.addMenu('&Edit')
        fileMenu.addAction(open_file)
        fileMenu.addAction(close_file)
        editMenu.addAction(change_detector)
        editMenu.addAction(change_additional_tracker)

        # выстраивание разметки приложения
        self.grid = QGridLayout()
        
        self.displaying_classes_layout = QVBoxLayout()
        self.horizontal_layout = QHBoxLayout()
        self.bboxes_display_control_layout = QVBoxLayout()
        self.control_layout = QVBoxLayout()
        self.prev_next_layout = QHBoxLayout()
        self.bboxes_display_control_layout.addWidget(self.is_logging_checkbox)
        self.bboxes_display_control_layout.addWidget(bboxes_correction_info)
        self.bboxes_display_control_layout.addWidget(show_auto_bboxes_button)
        self.bboxes_display_control_layout.addWidget(show_tracked_button)
        self.bboxes_display_control_layout.addWidget(show_tracked_and_raw_button)
        self.bboxes_display_control_layout.addStretch(1)

        self.prev_next_layout.addWidget(previous_frame_button)
        self.prev_next_layout.addWidget(next_frame_button)

        self.control_layout.addWidget(self.current_detector_label)
        self.control_layout.addWidget(self.current_additional_tracker_label)
        self.control_layout.addWidget(self.current_frame_label)
        self.control_layout.addWidget(self.current_frame_display)
        self.control_layout.addWidget(self.frames_num_label)
        self.control_layout.addWidget(self.all_frames_display)
        self.control_layout.addWidget(reset_tracker_and_set_frame_button)

        self.control_layout.addWidget(self.disable_add_tracking)
        #self.control_layout.addWidget(autoplay_button)
        self.control_layout.addLayout(self.prev_next_layout)

        self.displaying_classes_layout.addWidget(self.classes_with_description_table)
        self.displaying_classes_layout.addWidget(register_objects_button)
        self.displaying_classes_layout.addWidget(delete_registered_objects_button)
        self.displaying_classes_layout.addWidget(associate_auto_bboxes_button)
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
        self.is_autoplay = False
        # обнуление таблицы с отслеживаемыми объектами
        try:
            self.frame_with_boxes.bboxes_container.reset_tracking_objects_table()
        except:
            pass

    def set_all_params_to_default(self):
        # объект для перехода между кадрами видео и чтения кадров
        self.video_capture = None
        # путь до папки, где хранится видео и папка, куда записываются рамки
        self.path_to_labelling_folder = None
        self.paths_to_labels_list = []
        self.path_to_video = None
        self.window_name = None
        self.frame_with_boxes = None
        self.img_rows = None
        self.img_cols = None

        self.is_autoplay = False

        self.set_tracking_params_to_default()

        self.autosave_mode = False

        #self.automatic_bboxes_container = BboxesContainer(self.)

        #self.reset_table()

    def change_additional_tracker_handling(self):
        select_add_tracker_dialog = SelectModelDialog(list(self.additional_trackers_create_functions_dict.keys()))
        is_changed = select_add_tracker_dialog.exec()
        self.is_autoplay = False
        if not is_changed:
            return
        else:
            self.additional_tracker_type = select_add_tracker_dialog.current_model
            self.current_additional_tracker_label.setText(f'Текущий доп. трэкер: {self.additional_tracker_type}')
            if self.video_capture is None:
                return
            else:
                ret = show_info_message_box(
                    'ВНИМАНИЕ!',
                    'При смене доп. трекера процесс отслеживания обнуляется и начинается заново! Снимается выделение со всех отслеживаемых объектов, их придется выбирать заново. Выполнить?',
                    QMessageBox.Yes|QMessageBox.No,
                    QMessageBox.Warning
                )
                if ret == QMessageBox.No:
                    return
                else:
                    self.reset_additional_trackers()
                    self.update_objects_descr_table()

                    self.read_frame()
    
    def change_detector_handling(self):
        #models_names_list = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
        select_detector_dialog = SelectModelDialog(self.detectors_names_list)
        is_changed = select_detector_dialog.exec()
        self.is_autoplay = False
        if not is_changed:
            return
        else:
            self.tracker_type = f'{select_detector_dialog.current_model}.pt'
            self.current_detector_label.setText(f'Текущий НС детектор: {self.tracker_type.split(".")[0]}')
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
                    self.reset_tracker()
                    #!!!!!
                    self.frame_with_boxes.bboxes_container.unregister_all_bboxes() 
                    self.update_objects_descr_table()

                    self.read_frame()

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
        
        set_frame_dialog = SetFrameIdxDialog(self.frame_number)
        set_frame_dialog.exec()
        if set_frame_dialog.frame_idx is None:
            return
        
        self.current_frame_idx = set_frame_dialog.frame_idx

        self.unselect_all_table_items()
        self.set_tracking_params_to_default()
        
        self.reset_tracker()
        self.reset_additional_trackers()
        self.update_objects_descr_table()

        self.read_frame()

    def show_auto_bboxes_button_handling(self):
        if self.frame_with_boxes is None:
            return
        self.frame_with_boxes.bboxes_container.change_bboxes_displaying_type(displaying_type='auto')

    def show_regstered_and_auto_bboxes_button_handling(self):
        if self.frame_with_boxes is None:
            return
        self.frame_with_boxes.bboxes_container.change_bboxes_displaying_type(displaying_type='full')

    def show_registered_bboxes_button_handling(self):
        if self.frame_with_boxes is None:
            return
        self.frame_with_boxes.bboxes_container.change_bboxes_displaying_type(displaying_type='registered')

    def cancell_register_objects_button_handling(self):
        if self.video_capture is None:
            return
        
        ret = show_info_message_box(
                'ВНИМАНИЕ!',
                'Данное действие удалит все зарегистрированные рамки. Продолжить?',
                QMessageBox.Yes|QMessageBox.No,
                QMessageBox.Warning)
        self.is_autoplay = False
        if ret == QMessageBox.No:
            return
        
        self.reset_tracker()
        self.reset_additional_trackers()
        self.set_tracking_params_to_default()

        self.read_frame()

    def add_new_person_handling(self):
        self.is_autoplay = False
        if len(self.obj_descr2registered_bbox_dict) == 0:
            show_info_message_box(
                'Ошибка загрузки описаний объектов',
                'Сначала загрузите видео с описением классов',
                QMessageBox.Ok,
                QMessageBox.Critical
                )
            return
        
        if len(self.frame_with_boxes.bboxes_container) == 0:
            show_info_message_box(
                'Ошибка рамок объектов',
                'На видео не обнаружены объекты',
                QMessageBox.Ok,
                QMessageBox.Critical)
            
            return
        
    def associate_auto_bboxes_handling(self):
        '''
        Обработчик кнопки ассоциирования автоматически сгененрированных рамок и 
        '''
        self.is_autoplay = False
        if self.video_capture is None:
            return
        
        associate_auto_bboxes_dialog = AssociateRegisteredAndAutoBboxesDialog(
            registered_objects_container=self.frame_with_boxes.bboxes_container
        )
        associate_auto_bboxes_dialog.exec()

        current_class_name = associate_auto_bboxes_dialog.current_class_name
        current_auto_idx = associate_auto_bboxes_dialog.current_auto_idx
        current_registered_idx = associate_auto_bboxes_dialog.current_registered_idx
        current_registered_object_descr = associate_auto_bboxes_dialog.current_registered_object_descr

        if current_class_name == '---' or current_registered_object_descr == '---':
            return
        
        # Ищем рамку того же самого зарегистрированного объекта, если он есть в таблице
        existent_bbox = self.frame_with_boxes.bboxes_container.find_bbox(class_name=current_class_name, registered_idx=current_registered_idx, object_description=current_registered_object_descr)

        if len(existent_bbox) == 1:
            existent_bbox = existent_bbox['bbox'].values[0]
            self.frame_with_boxes.bboxes_container.pop(existent_bbox)

        path_to_db = glob.glob(os.path.join(f"{self.settings_dict['last_opened_folder']}", "*.csv"))[0]
        self.frame_with_boxes.bboxes_container.assocoate_bbox_with_registered_object(
            class_name=current_class_name,
            auto_idx=current_auto_idx,
            object_description=current_registered_object_descr,
            registered_idx=current_registered_idx
        )
        self.update_objects_descr_table()

    def delete_registered_handling(self):
        '''
        Обработчик кнопки удаления нового объекта
        '''
        self.is_autoplay = False
        if self.video_capture is None:
            return
        
        delete_registered_dialog = DeleteTrackingObjectsDialog(
            registered_objects_container=self.frame_with_boxes.bboxes_container
        )
        delete_registered_dialog.exec()

        deleting_class_name = delete_registered_dialog.current_class_name
        deleting_class_idx = delete_registered_dialog.current_class_idx
        deleting_object_descr = delete_registered_dialog.current_object_descr
        
        if deleting_class_name == '---':
            return
        
        
        path_to_db = glob.glob(os.path.join(f"{self.settings_dict['last_opened_folder']}", "*.csv"))[0]
        self.frame_with_boxes.bboxes_container.delete_from_tracking_objects_db(
            deleting_class_name, deleting_class_idx, deleting_object_descr, path_to_db)

        self.update_objects_descr_table()

    def register_objects_handling(self):
        '''
        Обработчик кнопки регистрации нового объекта
        '''
        self.is_autoplay = False
        if self.video_capture is None:
            return

        # вызов диалогового окна позволяет изменить имя класса всего для одной рамки
        register_persons_dialog = RegisterTrackingObjectsDialog(
            registered_objects_container=self.frame_with_boxes.bboxes_container,
            available_classes_list=list(self.tracker.tracker.names.values()))
        register_persons_dialog.exec()

        current_class_name = register_persons_dialog.current_class_name
        current_object_descr = register_persons_dialog.current_object_descr
        #current_auto_bbox = register_persons_dialog.current_auto_bbox

        # если имя класса осталось по умолчанию, то выходим
        if current_class_name == '---':
            return -1 # надо для того, чтобы в методе self.check_registered_in_disappeared_bboxes просигнализировать, что мы перерегистрировали рамку

        # добавляем объект в БД отслеживаемых объектов
        path_to_db = glob.glob(os.path.join(f"{self.settings_dict['last_opened_folder']}", "*.csv"))[0]
        self.frame_with_boxes.bboxes_container.append_to_tracking_objects_db(current_class_name, current_object_descr, path_to_db)

        self.update_objects_descr_table()

    def keyPressEvent(self, event):
        '''
        Реализация перехода между кадрами посредством клавиш на клавиатуре "<", ">" в разных раскладках,
        а также авторовоспроизведения 30 кадров посредством клавиши "]"
        '''
        if event.text() == '.' or event.text().lower() == 'ю':
            self.next_frame_button_handling()
            return
        elif event.text() == ',' or event.text().lower() == 'б':
            self.previous_frame_button_handling()
            return
        
        if event.text().lower() == ']' or event.text().lower() == 'ъ':
            #self.autoplay()
            return
    
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
            self.classes_with_description_table.item(row_idx, 2).setSelected(False)
            self.classes_with_description_table.item(row_idx, 3).setSelected(False)

    def table_cell_click_handling(self, row, col):
        item = self.classes_with_description_table.item(row, col)    

    def reinit_add_trackers_for_all_add_tracked_bboxes(self):
        '''
        Переинициализация координат всех рамок, для которых выполняется доп. трекинг
        '''
        additionaly_tracked_and_registered_df\
            = self.frame_with_boxes.bboxes_container.get_all_additionaly_tracked_registered_bboxes()
        for index, row in additionaly_tracked_and_registered_df.iterrows():
            bbox = row['bbox']
            self.reinit_additional_tracker_for_bbox(self.frame_with_boxes.img, bbox)
            
    def disable_add_tracking_slot(self):
        '''
        Обработчик checkbox, отвечающего за автоматическое сохранение кадра при переходе на новый
        '''
        if not self.disable_add_tracking.isChecked():
            # если галочка не нажата, то переинициализируем все рамки для доп трекеров
            self.reinit_add_trackers_for_all_add_tracked_bboxes()
    
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
        Метод, выполняющий обработку сигнала о создании рамки вручную
        '''
        self.is_autoplay = False
        # ищем вновь созданную рамку по имени класса ('?'), автоматическому индексу (-1) и "ручному" индексу (-1)
        manually_created_bbox_df = self.frame_with_boxes.bboxes_container.find_bbox(class_name='?', auto_idx=-1, registered_idx=-1)        
        manually_created_bbox = manually_created_bbox_df['bbox'].values[0]
        
        # вызываем новое диалоговое окно, чтобы выбрать имя класса для рамки        
        label_new_bbox_dialog = LabelNewBoxDialog(
            registered_objects_container=self.frame_with_boxes.bboxes_container)
        label_new_bbox_dialog.exec()
        
        current_class_name = label_new_bbox_dialog.current_class_name
        current_object_descr = label_new_bbox_dialog.current_object_descr
        current_registered_idx = label_new_bbox_dialog.current_registered_idx

        # удаляем вновь созданную рамку с именем класса "?" и индексами объектов, равными -1
        self.frame_with_boxes.bboxes_container.pop(manually_created_bbox)

        if current_class_name == '---':
            # если имя класса не выбрано, то надо рисовать заново
            return
        
        manually_created_bbox.class_name = current_class_name
        manually_created_bbox.registered_idx = current_registered_idx
        manually_created_bbox.color = (0, 255, 0)

        # добавляем созданную вручную рамку
        self.frame_with_boxes.bboxes_container.update_bbox(manually_created_bbox)
        # ассоциируем созданную вручную рамку и описание объекта
        self.frame_with_boxes.bboxes_container.assocoate_bbox_with_registered_object(
            class_name=current_class_name,
            auto_idx=manually_created_bbox.auto_idx,
            object_description=current_object_descr,
            registered_idx=current_registered_idx
        )

        self.update_objects_descr_table()



    def reset_additional_trackers(self):
        # обнуляем все трекеры
        self.additional_trackers_dict = {}#[bbox_name] = self.additional_trackers_create_functions_dict[self.additional_tracker_type]

    def reset_tracker(self):
        '''
        Обнуление трекера, чтобы начать процесс детекции и трекинга заново
        '''
        self.tracker = YoloTracker(model_type=self.tracker_type)

    def update_bboxes_on_frame(self):
        '''
        Этот метод вызывается, когда выполняется ручная коррекция рамки в кадре
        '''
        self.bbox_after_correction = self.frame_with_boxes.bbox_after_correction

        if self.is_logging_checkbox.isChecked():
                # сохраняем информацию о рамках до всяческой коррекции
                self.bboxes_container_after_corrections.update_bbox(self.bbox_after_correction)

        if not self.disable_add_tracking.isChecked():
            self.reinit_add_trackers_for_all_add_tracked_bboxes()   
            
    def reset_display(self):
        #Обнуление значений на экране и на слайдере
        self.set_display_value(0)
        self.all_frames_display.display(0)
    
    def set_display_value(self, val):
        #self.frame_slider.setValue(val)
        self.current_frame_display.display(val)
    
    def close_video(self):
        '''
        Обработчик закрытия файла
        Сохраняет рамки, закрывает поток чтения изображения и делает объект изображения с рамками пустым
        '''

        # закрываем поток, который отображает кадры видео
        self.close_imshow_thread()
        # обнуляем все праметры
        self.set_all_params_to_default()
        # обновляем трекер
        self.reset_table()
        self.reset_tracker()
        self.reset_additional_trackers()
    
    def get_object_descr_list(self):
        '''
        Получение списка описаний объектов. Каждый эл-т списка составляет кортеж из четырех элементов:
        индекс зарегистрированного объекта, имя класса, описание объекта, имя автоматической рамки
        '''
        objects_descr_list = []
        for i, row in self.frame_with_boxes.bboxes_container.registered_objects_db.sort_values(by=['class_name', 'object_idx']).iterrows():
            registered_idx = row['object_idx']
            class_name = row['class_name']
            object_descr = row['object_description']
            auto_bbox_name = ''
            # ищем автоматическую рамку по 
            bboxes = self.frame_with_boxes.bboxes_container.get_auto_bbox_from_registered(class_name, registered_idx, object_descr)
            
            if len(bboxes) == 1:
                auto_bbox_name = f'{class_name},(AG){bboxes["auto_idx"].values[0]}'

            objects_descr_list.append((registered_idx, class_name, object_descr, auto_bbox_name))
        return objects_descr_list

    def update_objects_descr_table(self):
        objects_descr_list = self.get_object_descr_list()
        self.classes_with_description_table.setRowCount(len(objects_descr_list))
        for i, (registered_idx, class_name, obj_descr, auto_bbox_name) in enumerate(objects_descr_list):
            self.classes_with_description_table.setItem(i, 0, QTableWidgetItem(f'{registered_idx}'))
            self.classes_with_description_table.setItem(i, 1, QTableWidgetItem(f'{class_name},{registered_idx}'))
            self.classes_with_description_table.setItem(i, 2, QTableWidgetItem(obj_descr))
            self.classes_with_description_table.setItem(i, 3, QTableWidgetItem(auto_bbox_name))

    def read_tracking_objects_db(self, path_to_dir, name):
        '''
        Чтение базы данных, содержащей информацию о файлах
        '''
        name = '.'.join(name.split('.')[:-1])
        path_to_db = os.path.join(path_to_dir, f'{name}.csv')
        if os.path.isfile(path_to_db):
            database = pd.read_csv(path_to_db)
            database = database.fillna(value='')
        else:
            database = pd.DataFrame(columns=['object_idx', 'class_name', 'object_description'])
            database.to_csv(path_to_db, index=False)
        return database
        
    def open_file(self):
        # закрываем поток, который отображает кадры видео
        self.close_imshow_thread()
        # обнуляем все праметры
        self.set_all_params_to_default()
        # обновляем трекер
        self.reset_table()
        self.reset_tracker()

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

        # обновляем путь до последнего открытого файла и перезаписываем файл конфигурации
        self.settings_dict['last_opened_folder'] = path_to_folder
        with open('settings.json', 'w', encoding='utf-8') as fd:
            json.dump(self.settings_dict, fd)

        # читаем базу данных с объектами, которые надо отслеживать
        registered_objects_db = self.read_tracking_objects_db(path_to_folder, name)

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

        # отображаем общее количество кадров на специальном индикаторе
        self.all_frames_display.display(self.frame_number)

        # выясняем размер (кол-во строк и столбцов) кадра
        self.img_rows, self.img_cols = frame.shape[:2]

        # выставляем счетчик кадров: если уже были сформированы рамки, то счетчик кадров делаем равным номеру последнего кадра 
        if len(self.paths_to_labels_list) > 0:
            self.current_frame_idx = len(self.paths_to_labels_list) - 1
        else:
            self.current_frame_idx = 0
        
        # Записываем имя окна
        self.window_name = name
        
        # создаем объект BboxFrameTracker, позволяющий отображать и изменять локализационные рамки на кадре видео
        self.frame_with_boxes = BboxFrameTracker(img=frame, registered_objects_db=registered_objects_db)

        # обновляем таблицу, где отображены все отслеживаемые объекты
        self.update_objects_descr_table()
        
        # Подготавливаем и запускаем отдельный поток, в котором будет отображаться кадр с рамками
        # и будут изменяться рамки
        self.setup_imshow_thread()
        self.imshow_thread.setup_frame(self.frame_with_boxes, self.window_name)
        self.imshow_thread.start()

        # сразу открываем видео
        self.read_frame()

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
        
        # создаем список рамок зарегистрированных отслеживаемых объектов
        registered_bboxes_list = self.frame_with_boxes.bboxes_container.get_all_registered_bboxes_list()
        # обновляем словарь новыми рамками
        labels_json_dict.update(
            {f'{bbox.class_name},{bbox.registered_idx}':[int(coord) for coord in bbox.coords] for bbox in registered_bboxes_list})
        
        # Сохраняем разметку
        with open(path_to_target_json_label, 'w', encoding='utf-8') as fd:
            json.dump(labels_json_dict, fd, indent=4)

        # сохранение лога
        if self.is_logging_checkbox.isChecked():
            path_to_logging_dir = os.path.join(self.settings_dict['last_opened_folder'], 'log')
            os.makedirs(path_to_logging_dir, exist_ok=True)
            path_to_bboxes_before_changes = os.path.join(
                path_to_logging_dir, f'{self.current_frame_idx:06d}.csv')
            self.bboxes_container_berfore_corrections.bboxes_df.to_csv(path_to_bboxes_before_changes, index=False)

    def next_frame_button_handling(self):
        if self.video_capture is None or self.frame_with_boxes is None:
            self.is_autoplay = False    
            if self.imshow_thread.isRunning():
                self.stop_imshow_thread()
            return
        if not self.is_logging_checkbox.isChecked():
            show_info_message_box(
                window_title='Логгирование отключено',
                info_text='Логгирование отключено. Пожалуйста нажмите галочку "ВКЛЮЧИТЬ ЗАПИСЬ ЛОГА", чтобы продолжить воспроизведение',
                buttons=QMessageBox.Ok,
                icon_type=QMessageBox.Critical)
            return
        # сохраняем все рамки
        if self.current_frame_idx > -1:
            pass
        
        # сохраняем рамки
        self.save_labels()

        self.current_frame_idx += 1
        if self.current_frame_idx >= self.frame_number:
            self.is_autoplay = False
            self.current_frame_idx = self.frame_number - 1
            show_info_message_box('Конец видео', 'Вы достигли конца видео', QMessageBox.Ok, QMessageBox.Information)
            return

        self.read_frame()

        return

    def previous_frame_button_handling(self):
        if self.video_capture is None or self.frame_with_boxes is None:
            self.is_autoplay = False
            if self.imshow_thread.isRunning():
                self.stop_imshow_thread()
            return
        
        if not self.is_logging_checkbox.isChecked():
            show_info_message_box(
                window_title='Логгирование отключено',
                info_text='Логгирование отключено. Пожалуйста нажмите галочку "ВКЛЮЧИТЬ ЗАПИСЬ ЛОГА", чтобы продолжить воспроизведение',
                buttons=QMessageBox.Ok,
                icon_type=QMessageBox.Critical)
            return
        
        # сохраняем рамки
        self.save_labels()

        self.current_frame_idx -= 1
        if self.current_frame_idx < 0:
            self.is_autoplay = False
            return
               
        self.read_frame()

    def autoplay(self):
        self.is_autoplay = True
        for i in range(30):
            if not self.is_autoplay:
                break
            self.next_frame_button_handling()

    def try_alternative_tracking(self):
        # 1. Ищем рамки, для которых надо выполнить альтернативный трекинг
        # признак таких рамок - auto_idx == -1 AND registered_idx != -1
        alternative_tracking_bboxes_filter_condidion = (self.frame_with_boxes.bboxes_container.bboxes_df['auto_idx'] == -1)\
            & (self.frame_with_boxes.bboxes_container.bboxes_df['registered_idx'] != -1)
        alternative_tracking_bboxes_df = self.frame_with_boxes.bboxes_container.bboxes_df[alternative_tracking_bboxes_filter_condidion]

        # 2. Итерируем по рамкам
        for index, row in alternative_tracking_bboxes_df.iterrows():
            bbox = row['bbox']
            class_name = row['class_name']
            registered_idx = row['registered_idx']
            bbox_name = f'{class_name}(T),{registered_idx}'
            if self.disable_add_tracking.isChecked():
                # если включена галочка (QCheckbox) "Откл. доп трекинг", то мы не меняем координаты рамок
                self.frame_with_boxes.bboxes_container.update_existing_bbox_coords(index, bbox)
                continue
            
            # читаем координаты рамок из БД, полученные для предыдущего кадра
            bbox_coords = bbox.x0y0wh()
            try:
                # выполняем попытку трекинга
                _, tracked_bbox_coords = self.additional_trackers_dict[bbox_name].update(self.frame_with_boxes.img)
            except:
                # если трекер для текущей рамки не проинициализирован, то инициализируем его
                self.reinit_additional_tracker_for_bbox(self.frame_with_boxes.img, bbox)
                tracked_bbox_coords = bbox_coords
            
            tracked_bbox_coords = xywh2xyxy(*tracked_bbox_coords)    
            new_coords = process_box_coords(*tracked_bbox_coords, self.img_rows, self.img_cols)
            new_area = compute_bbox_area(*new_coords)

            if new_area < 16:    
                # если площадь рамки стала слишком маленькой, надо предупредить об этом
                # если с рамкой произошла какая-то беда, то оставляем предыдущие координаты
                new_coords = xywh2xyxy(*bbox_coords)

            bbox.coords = new_coords
            self.frame_with_boxes.bboxes_container.update_existing_bbox_coords(index, bbox)
    
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

    def read_frame(self):
        # проверка условий возможности чтения кадра: отсутствие объекта, отвечающего за чтение кадров видео
        # или номер текущего кадра превышает количество кадров в видео
        if self.video_capture is None or self.current_frame_idx >= self.frame_number:
            return
        
        if self.current_frame_idx < 0:
            self.current_frame_idx = 0
            return
        
        # устанавливаем значение дисплея, отображающего счетчик кадров
        self.set_display_value(self.current_frame_idx)
        
        # выставляем в объекте чтения кадров позицию текущего кадра, чтобы иметь возможность двигаться не только вперед, но и назад
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        # читаем текущий кадр
        ret, frame = self.video_capture.read()

        if ret:
            # обновляем отображаемые на видео рамки
            self.frame_with_boxes.update_img(frame)
            
            # выполняем трекинг
            try:
                yolo_predicted_bboxes_container = self.tracker.track(
                    bboxes_container=self.frame_with_boxes.bboxes_container,
                    source=frame,
                    persist=True,
                    verbose=True
                    )
            except:
                yolo_predicted_bboxes_container = self.frame_with_boxes.bboxes_container            
   
            # присваиваем объекту, обрабатывающему кадр с рамками, полученный из yolo bboxes_container
            self.frame_with_boxes.bboxes_container = yolo_predicted_bboxes_container
            disappeared_bboxes = self.frame_with_boxes.bboxes_container.check_updated_bboxes()
            if len(disappeared_bboxes) != 0:
                self.check_registered_in_disappeared_bboxes(disappeared_bboxes)

            if self.is_logging_checkbox.isChecked():
                # сохраняем информацию о рамках до всяческой коррекции
                registered_objects_db = self.frame_with_boxes.bboxes_container.registered_objects_db
                # обнуляем контейнер с измененными рамками
                self.bboxes_container_after_corrections = BboxesContainer(registered_objects_db)
                # копируем контейнер, полученный из автоматического трекера
                self.bboxes_container_berfore_corrections = deepcopy(self.frame_with_boxes.bboxes_container)

            self.try_alternative_tracking()
            
    
    def check_registered_in_disappeared_bboxes(self, disappeared_bboxes):
        '''
        Проверка, есть ли в пропавших рамках зарегистрированные и отслеживаемые объекты
        Если есть, то мы либо выполняем перерегистрацию отслеживаемого объекта на другие автоматические рамки
        либо выполняем дополнительный трекинг

        '''
        registered_bboxes_filter_condition = disappeared_bboxes['registered_idx'] != -1
        registered_bboxes = disappeared_bboxes[registered_bboxes_filter_condition]

        if len(registered_bboxes) == 0:
            return
        
        all_coordinates_df = self.frame_with_boxes.bboxes_container.get_all_bboxes_coordinates()
        for _, row in registered_bboxes.iterrows():
            disappeared_bbox = row['bbox']
            class_name = row['class_name']
            registered_idx = row['registered_idx']
            class_coords_df = all_coordinates_df[all_coordinates_df['class_name']==class_name].reset_index(drop=True)
            if len(class_coords_df) == 0:
                continue

            coord_matrix = class_coords_df[['x0', 'y0', 'x1', 'y1']].to_numpy()
            dist_arr = np.linalg.norm(coord_matrix-disappeared_bbox.numpy_coords(), axis=1)
            nearest_idx = dist_arr.argmax()
            auto_idx = class_coords_df.loc[nearest_idx]['auto_idx']
            disappeared_bbox_name = f'{class_name}(T),{registered_idx}'
            nearest_name = f'{class_name}(AG),{auto_idx}'
            # сначала пытаемся зарегистрировать наиболее близкую рамку
            result = show_info_message_box(
                'Отслеживаемый объект пропал',
                f'Пропал объект {disappeared_bbox_name}. Наиболее похожая автоматически сгенерированная рамка: {nearest_name}. Выполнить ее ассоциацию?',
                QMessageBox.Yes|QMessageBox.No,
                QMessageBox.Warning
            )
            self.is_autoplay = False
            if result == QMessageBox.Yes:
                ret = self.associate_auto_bboxes_handling()
                if ret is None:
                    continue

            # потом пытаемся выполнить альтернативный трекинг
            result = show_info_message_box(
                    'Отслеживаемый объект пропал',
                    f'Пропал объект {disappeared_bbox_name}. Запустить дополнительный трекер?',
                    QMessageBox.Yes|QMessageBox.No,
                    QMessageBox.Warning
                )
            if result == QMessageBox.Yes:
                # делаем рамку восприимчивой к альтернативному трекингу и добавляем ее в таблицу
                disappeared_bbox.auto_idx = -1
                disappeared_bbox.displaying_type = 'registered'
                self.frame_with_boxes.bboxes_container.update_bbox(disappeared_bbox)
                
                # читаем предыдущий кадр, т.к. рамка есть только для объекта на предыдущем кадре, а его положение могло измениться
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx-1)
                _, prev_frame = self.video_capture.read()
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                # инициализируем трекер
                self.reinit_additional_tracker_for_bbox(prev_frame, disappeared_bbox)

    def reinit_additional_tracker_for_bbox(self, frame, bbox):
        '''
        Выполнение инициализации дополнительного трекера. Если трекер уже существует, то новый трекер затирает существующий
        '''
        xywh_bbox_coords = bbox.x0y0wh()
        bbox_name = f'{bbox.class_name}(T),{bbox.registered_idx}'
        additional_tracker = self.additional_trackers_create_functions_dict[self.additional_tracker_type]()
        additional_tracker.init(frame, xywh_bbox_coords)        
        self.additional_trackers_dict[bbox_name] = additional_tracker

    def stop_showing(self):
        '''
        Прекращение демонстрации кадров видео в окне opencv
        '''
        if self.is_showing:
            self.is_showing = False
            cv2.destroyAllWindows()

class YoloTracker:
    def __init__(self, model_type='yolov8n.pt'):
        '''
        Класс-обертка для трэкера. Нужна для того, чтобы совместить выполнение трекинга и присвоение имен классов детектируемых объектов
        '''
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        print(f'We are using {device} device for tracking')
        print()
        self.tracker = YOLO(model_type).to(device)
        # словарь для перевода имени класса в номер метки
        # self.tracker.names - словарь, содержащий пары {индекс: имя_класса}
        self.name2class_idx = {val: key for key, val in self.tracker.names.items()}


    def track(self, bboxes_container, *yolo_args, **yolo_kwargs):
        '''
        target_class_name - имя класса, который мы собираемся детектировать
        Return: 
            bboxes_container - словарь, который хранит 
        '''
        # выполнение трекинга 
        results = self.tracker.track(*yolo_args, **yolo_kwargs)[0]

        # получение координат рамок
        bboxes = results.boxes.xyxy.long().numpy()
        # получение индексов объектов
        ids = results.boxes.id.long().numpy()
        
        # получение списка детектированных классов
        detected_classes = [self.tracker.names[cls_idx] for cls_idx in results.boxes.cls.long().numpy()]
        
        # строки и столбцы изображения нужны для создания объектов рамок
        img_rows, img_cols = results[0].orig_img.shape[:-1]
        
        # этот параметр нужен, чтобы рамка строилась не впритык объекту, а захватывала еще некоторую дополнительную область
        bbox_append_value = int(min(img_rows, img_cols)*0.025)

        for bbox, id, class_name in zip(bboxes, ids, detected_classes):
            x0,y0,x1,y1 = bbox
            # добавляем несколько пикселей, чтобы рамка строилась не впритык к объекту
            x0,y0,x1,y1 = x0 - bbox_append_value, y0 - bbox_append_value, x1 + bbox_append_value, y1 + bbox_append_value
            # делаем так, чтобы рамка не выходила за пределы кадра
            x0,y0,x1,y1 = process_box_coords(x0,y0,x1,y1, img_rows, img_cols)
            # пока что оставляем всего лишь один цвет - черный
            color = (0,0,0)

            bbox = Bbox(
                x0, y0, x1, y1,
                img_rows, img_cols,
                class_name=class_name,
                auto_idx=id,
                registered_idx=-1,
                object_description='',
                color=color,
                displaying_type='auto'
                )            

            bboxes_container.update_bbox(bbox)

        return bboxes_container

class ImshowThread(QThread):
    '''
    Отображение кадров запускается в отдельном потоке, который сигнализирует
    основному потоку об изменениях рамок (коррекция координат углов, перетаскивание, создание и удаление) на кадре видео
    '''
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