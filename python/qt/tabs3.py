#!/usr/bin/python3
#\file    tabs3.py
#\brief   Tab test 3: enable/disable tab.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.20, 2023
import sys
from PyQt4 import QtCore,QtGui

def Print(*s):
  for ss in s:  print(ss, end=' ')
  print('')

class TTab(QtGui.QWidget):
  def __init__(self):
    QtGui.QWidget.__init__(self)
    self.InitUI()

  def InitUI(self):
    # Set window size.
    self.resize(320, 240)

    # Set window title
    self.setWindowTitle('Tab3')

    layout= QtGui.QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)

    # Create tabs
    tabs= QtGui.QTabWidget()
    tab_names= ['Tab 1','Tab 2','Tab 3']
    tab_name_to_widget= {}
    tab_name_to_index= {}
    for tab_idx,tab_name in enumerate(tab_names):
      tab= QtGui.QWidget()
      tabs.addTab(tab,tab_name)
      tab_name_to_widget[tab_name]= tab
      tab_name_to_index[tab_name]= tab_idx
    set_current_tab= lambda tab_name:tabs.setCurrentIndex(tab_name_to_index[tab_name])

    # Resize width and height
    tabs.resize(250, 150)

    # Set layout of first tab
    vBoxlayout= QtGui.QVBoxLayout()

    btn1= QtGui.QPushButton('To go Tab 2')
    btn1.clicked.connect(lambda b: set_current_tab('Tab 2'))
    vBoxlayout.addWidget(btn1)

    btn2= QtGui.QPushButton('To go Tab 3')
    btn2.clicked.connect(lambda b: set_current_tab('Tab 3'))
    vBoxlayout.addWidget(btn2)

    btn3= QtGui.QPushButton('Disable Tab 2')
    btn3.setCheckable(True)
    btn3.setChecked(True)
    btn3.clicked.connect(lambda checked,btn3=btn3:
                           (btn3.setText('Disable Tab 2'),
                            tabs.setTabEnabled(tab_name_to_index['Tab 2'],True),
                            #tab_name_to_widget['Tab 2'].setEnabled(checked)
                            ) if checked else
                           (btn3.setText('Enable Tab 2'),
                            tabs.setTabEnabled(tab_name_to_index['Tab 2'],False),
                            #tab_name_to_widget['Tab 2'].setEnabled(checked)
                            ))
    #btn3.clicked.connect(lambda b,btn3=btn3: Print('Enabled') if btn3.isChecked() else Print('Disabled'))
    #btn3.toggled.connect(lambda checked,btn3=btn3: btn3.setText('Disable Tab 2') if checked else btn3.setText('Enable Tab 2'))
    vBoxlayout.addWidget(btn3)

    tab_name_to_widget['Tab 1'].setLayout(vBoxlayout)

    layout.addWidget(tabs)

    set_current_tab('Tab 1')

    self.setLayout(layout)

    # Show window
    self.show()

# Create an PyQT4 application object.
a = QtGui.QApplication(sys.argv)

# The QWidget widget is the base class of all user interface objects in PyQt4.
w = TTab()

sys.exit(a.exec_())

