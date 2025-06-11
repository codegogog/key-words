# !/usr/bin/python3
# -*- coding: utf-8 -*-

import uploadFile
import textPrecessing
import statistics
import outPut
import intermediate
from collections import defaultdict
import os
import sys
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import file_processor



# 归一化
def normalized(wordsData, dict):
    minWord = min(dict, key=dict.get)
    minValue = dict[minWord]
    maxWord = max(dict, key=dict.get)
    maxValue = dict[maxWord]
    diff = maxValue - minValue
    for word in wordsData:
        dict[word] = (dict[word] - minValue) / diff
    return dict

# 计算词语得分
def calculateScore(wordsData, interDensity, wordsLoc, wordsTfidf, wordsFlagWeight):
    score = defaultdict(float)
    sW = 0.4  # 降低居间度权重
    Tw = 0.6  # 提高统计特征权重
    posW = 0.3  # 降低词性权重
    locW = 0.8  # 降低位置权重
    tfidfW = 1.2  # 提高TF-IDF权重
    
    for word in wordsData:
        # 使用非线性组合方式
        score[word] = (sW * interDensity[word] + 
                      Tw * (posW * wordsFlagWeight[word] + 
                           locW * wordsLoc[word] + 
                           tfidfW * wordsTfidf[word])) * \
                     (1 + 0.2 * wordsLoc[word])  # 位置特征作为修正因子

    return score


def getFileName(path):
    file_name = os.path.basename(path)
    file_path = os.path.dirname(path)
    return file_name, file_path

# 处理前后继关系
def addWord(interDensity, wordsData, wordsFlagDict, nextDict, nextWordSum, preDict, preWordSum):
    for word1 in nextDict:
        if nextWordSum[word1] > 2:
            for word2 in nextDict[word1]:
                if word1 in preDict[word2] and nextDict[word1][word2] / nextWordSum[word1] >= 0.8 and preDict[word2][word1] / preWordSum[word2] >= 0.8:
                    strs = word1 + word2
                    wordsData.append(strs)
                    interDensity[strs] = max(interDensity[word1], interDensity[word2])
                    wordsFlagDict[strs] = wordsFlagDict[word2]
    return interDensity, wordsData, wordsFlagDict

class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.current_files = []  # 存储当前处理的文件列表
        self.file_processor = file_processor.FileProcessor()  # 创建文件处理器实例
        self.use_simple_preprocess = True  # 默认使用简化预处理

    def initUI(self):
        # 设置窗口样式
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                font-family: 'Microsoft YaHei', Arial;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QLabel {
                font-size: 14px;
                color: #333333;
                padding: 5px;
            }
            QProgressBar {
                border: 2px solid #4CAF50;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f0f0;
                height: 25px;
                font-size: 13px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 8px;
                background-color: white;
                font-size: 14px;
            }
            QListWidget {
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: white;
                font-size: 13px;
            }
            QListWidget::item {
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #e0f2e0;
                color: #333333;
            }
        """)

        # 主布局
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # 标题
        title_label = QLabel("中文关键词提取系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #333333; margin: 20px; padding: 10px;")
        main_layout.addWidget(title_label)

        # 文件列表区域
        file_list_layout = QVBoxLayout()
        file_list_label = QLabel("待处理文件列表：")
        file_list_label.setStyleSheet("font-weight: bold;")
        file_list_layout.addWidget(file_list_label)
        
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(150)
        file_list_layout.addWidget(self.file_list)
        
        main_layout.addLayout(file_list_layout)

        # 添加简化预处理的选择框
        preprocess_options_layout = QHBoxLayout()
        preprocess_options_label = QLabel("预处理方式：")
        preprocess_options_layout.addWidget(preprocess_options_label)
        
        self.preprocess_combo = QComboBox()
        self.preprocess_combo.addItem("简化预处理")
        self.preprocess_combo.addItem("完整预处理")
        self.preprocess_combo.setCurrentIndex(0)  # 默认选择简化预处理
        self.preprocess_combo.currentIndexChanged.connect(self.on_preprocess_option_changed)
        preprocess_options_layout.addWidget(self.preprocess_combo)
        preprocess_options_layout.addStretch()
        
        main_layout.addLayout(preprocess_options_layout)

        # 按钮网格布局
        grid = QGridLayout()
        grid.setSpacing(15)
        main_layout.addLayout(grid)

        # 设置按钮
        names = ['导入文档', '批量导入', '预处理', '提取语义特征', '提取统计特征', '计算词语得分']
        positions = [(0, j) for j in range(6)]
        for position, name in zip(positions, names):
            button = QPushButton(name)
            button.setObjectName(name)
            button.setEnabled(False)
            button.clicked.connect(self.on_click)
            grid.addWidget(button, *position)

        self.findChild(QPushButton, "导入文档").setEnabled(True)
        self.findChild(QPushButton, "批量导入").setEnabled(True)

        # 状态标签
        status_group = QGroupBox("处理状态")
        status_layout = QVBoxLayout()
        self.status_labels = {}
        for name in names:
            label = QLabel("")
            label.setObjectName(name)
            label.setAlignment(Qt.AlignLeft)
            label.setStyleSheet("color: #666666;")
            status_layout.addWidget(label)
            self.status_labels[name] = label

        status_group.setLayout(status_layout)
        grid.addWidget(status_group, 1, 0, 4, 6)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        grid.addWidget(self.progress_bar, 5, 0, 1, 6)

        # 查看按钮
        view_button = QPushButton('查看结果')
        view_button.setObjectName('查看')
        view_button.setEnabled(False)
        view_button.clicked.connect(self.on_click)
        grid.addWidget(view_button, 6, 2, 1, 2)

        # 结果显示区域
        result_label = QLabel("关键词提取结果：")
        result_label.setStyleSheet("font-weight: bold; margin-top: 15px;")
        main_layout.addWidget(result_label)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setVisible(True)
        self.result_text.setMinimumHeight(250)
        main_layout.addWidget(self.result_text)

        self.resize(1200, 900)  # 增加窗口大小
        self.center()
        self.setWindowTitle('中文关键词提取系统 ')
        self.show()

    # 窗口居中
    def center(self):

        # 获得窗口
        qr = self.frameGeometry()
        # 获得屏幕中心点
        cp = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def on_click(self):
        sender = self.sender()
        ButtonName = sender.text()
        
        try:
            if ButtonName == "导入文档":
                self.import_single_file()
            elif ButtonName == "批量导入":
                self.import_multiple_files()
            elif ButtonName == "预处理":
                self.process_files("预处理")
            elif ButtonName == "提取语义特征":
                self.process_files("提取语义特征")
            elif ButtonName == "提取统计特征":
                self.process_files("提取统计特征")
            elif ButtonName == "计算词语得分":
                self.process_files("计算词语得分")
            elif ButtonName == "查看结果":
                self.show_results()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理过程中发生错误：{str(e)}")

    def import_single_file(self):
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '', '文本文件 (*.txt);;所有文件 (*.*)')
        if openfile_name[0]:
            # 确保路径规范化
            file_path = os.path.normpath(openfile_name[0])
            self.current_files = [file_path]
            self.file_list.clear()
            self.file_list.addItem(file_path)
            self.update_buttons_state()

    def import_multiple_files(self):
        openfile_names = QFileDialog.getOpenFileNames(self, '选择文件', '', '文本文件 (*.txt);;所有文件 (*.*)')
        if openfile_names[0]:
            # 确保所有路径规范化
            self.current_files = [os.path.normpath(path) for path in openfile_names[0]]
            self.file_list.clear()
            for file in self.current_files:
                self.file_list.addItem(file)
            self.update_buttons_state()

    def update_buttons_state(self):
        """根据处理状态更新按钮状态"""
        has_files = len(self.current_files) > 0
        self.findChild(QPushButton, "预处理").setEnabled(has_files)
        self.findChild(QPushButton, "提取语义特征").setEnabled(False)
        self.findChild(QPushButton, "提取统计特征").setEnabled(False)
        self.findChild(QPushButton, "计算词语得分").setEnabled(False)
        self.findChild(QPushButton, "查看").setEnabled(False)
        
        # 如果文件已经处理过，自动启用对应按钮
        if has_files:
            try:
                for file_path in self.current_files:
                    file_name = os.path.splitext(os.path.basename(file_path))[0]
                    file_dir = os.path.dirname(file_path)
                    output_dir = os.path.normpath(os.path.join(file_dir, file_name + "_output"))
                    
                    # 检查预处理结果
                    if os.path.exists(os.path.join(output_dir, "预处理", "wordsData.txt")):
                        self.findChild(QPushButton, "提取语义特征").setEnabled(True)
                        
                    # 检查语义特征结果
                    if os.path.exists(os.path.join(output_dir, "语义特征", "interDensity.json")):
                        self.findChild(QPushButton, "提取统计特征").setEnabled(True)
                        
                    # 检查统计特征结果
                    if os.path.exists(os.path.join(output_dir, "统计特征", "wordsTfidf.json")):
                        self.findChild(QPushButton, "计算词语得分").setEnabled(True)
                        
                    # 检查关键词结果
                    if os.path.exists(os.path.join(output_dir, "词语得分", "keywords.txt")):
                        self.findChild(QPushButton, "查看").setEnabled(True)
            except Exception as e:
                print(f"检查处理状态时出错: {str(e)}")

    def update_progress(self, value, text):
        """更新进度条显示"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{text} {value}%")
        QApplication.processEvents()

    def on_preprocess_option_changed(self, index):
        """处理预处理选项变更"""
        self.use_simple_preprocess = (index == 0)
        print(f"预处理方式已切换为: {'简化预处理' if self.use_simple_preprocess else '完整预处理'}")

    def process_single_file(self, file_path, stage):
        """处理单个文件"""
        try:
            # 更新状态标签
            self.status_labels[stage].setText(f"正在处理文件：{os.path.basename(file_path)}")
            QApplication.processEvents()

            # 处理文件 - 针对预处理阶段，考虑用户选择的预处理方式
            if stage == "预处理":
                # 设置预处理模式
                self.file_processor.use_simple_preprocess = self.use_simple_preprocess
                
            # 处理文件
            result = self.file_processor.process_file(file_path, stage)

            # 更新状态标签
            self.status_labels[stage].setText(f"✓ 文件 {os.path.basename(file_path)} 处理完成")
            QApplication.processEvents()

            return result
        except Exception as e:
            self.status_labels[stage].setText(f"❌ 处理文件 {os.path.basename(file_path)} 时发生错误：{str(e)}")
            raise

    def process_files(self, stage):
        """处理多个文件"""
        if not self.current_files:
            QMessageBox.warning(self, "警告", "请先导入文件！")
            return

        self.update_progress(0, f"正在{stage}")
        QApplication.processEvents()

        try:
            for i, file_path in enumerate(self.current_files):
                self.process_single_file(file_path, stage)
                progress = int((i + 1) / len(self.current_files) * 100)
                self.update_progress(progress, f"正在{stage}... ({i+1}/{len(self.current_files)})")
                QApplication.processEvents()

            self.update_progress(100, f"{stage}完成！")
            
            # 更新按钮状态
            if stage == "预处理":
                self.findChild(QPushButton, "提取语义特征").setEnabled(True)
            elif stage == "提取语义特征":
                self.findChild(QPushButton, "提取统计特征").setEnabled(True)
            elif stage == "提取统计特征":
                self.findChild(QPushButton, "计算词语得分").setEnabled(True)
            elif stage == "计算词语得分":
                self.findChild(QPushButton, "查看").setEnabled(True)
                
            self.progress_bar.setVisible(False)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"{stage}过程中发生错误：{str(e)}")
            self.progress_bar.setVisible(False)

    def show_results(self):
        """显示处理结果"""
        if not self.current_files:
            QMessageBox.warning(self, "警告", "请先导入文件！")
            return

        self.result_text.setVisible(True)
        try:
            results = []
            for file_path in self.current_files:
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                file_dir = os.path.dirname(file_path)
                output_dir = os.path.normpath(os.path.join(file_dir, file_name + "_output"))
                keyword_path = os.path.join(output_dir, "词语得分", "keywords.txt")
                
                if os.path.exists(keyword_path):
                    with open(keyword_path, 'r', encoding='utf-8') as f:
                        results.append(f"文件：{os.path.basename(file_path)}\n{f.read()}\n")
                else:
                    results.append(f"文件：{os.path.basename(file_path)} - 未找到结果文件\n")

            self.result_text.setText("\n".join(results))
        except Exception as e:
            self.result_text.setText(f"读取结果文件时发生错误: {str(e)}")


if __name__ == '__main__':
    # 创建应用程序和对象
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())