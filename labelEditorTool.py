import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtCore
from viewer.viewer import Viewer
from PyQt5.QtGui import QImage, QPixmap
from dataset.kitti_dataset import KittiDetectionDataset

class ImageWindow(QtWidgets.QWidget):
    def __init__(self, width=900, height=300, parent=None, x=900, y=790):
        super().__init__(parent)
        self.setWindowTitle("Front view")
        self.setFixedSize(width, height)
        # Set initial position
        self.move(x, y)
        
        # Set the window to behave as a separate top-level window
        self.setWindowFlag(QtCore.Qt.Window)

        # Set up the QLabel to display the image
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setFixedSize(width, height)
        
        # Layout for image display
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def display_image(self, image=None):
        
        """ Load and display the image in the QLabel.
        
        Parameters:
        - image (numpy array): OpenCV image array.
        - image_path (str): Optional file path for the image. """
       
        if image is not None:
            # Convert the OpenCV image (numpy array) to QImage
            height, width, channels = image.shape
            bytes_per_line = channels * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            # Scale pixmap to the QLabel's fixed size
            scaled_pixmap = pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        
        else:
            # Clear the label if no image is available
            self.image_label.clear()

class LabelEditor(QtWidgets.QWidget):
    def __init__(self, dataset_path, label_path, width=400, height=400):
        super().__init__()
        
        # Initialize dataset and viewer
        self.dataset = KittiDetectionDataset(dataset_path, label_path)
        self.viewer = Viewer(box_type="Kitti", window_size=(800, 950))
        self.viewerR = Viewer(box_type="Kitti", window_size=(900, 300), pos=(900,50))
        self.viewerL = Viewer(box_type="Kitti", window_size=(900, 300), pos=(900,420))
        self.image_window = ImageWindow(parent=self)
        self.rows_to_delete = set()
        self.viewer.set_ob_color_map('gnuplot')
        self.viewerR.set_ob_color_map('gnuplot')
        self.viewerL.set_ob_color_map('gnuplot')
        self.label_path_of_current_sample=None

        self.resize(width, height)
        
        self.current_index = 0

        # Set up GUI layout
        self.setWindowTitle("Dataset Viewer and Editor")
        layout = QtWidgets.QVBoxLayout()

        # Navigation buttons
        self.file_label = QtWidgets.QLabel()
        self.empty_file_message = QtWidgets.QLabel()

        text_messages = QtWidgets.QHBoxLayout()
        text_messages.addWidget(self.file_label)  # Add Previous button to the left
        text_messages.addStretch()
        text_messages.addWidget(self.empty_file_message)

        self.prev_button = QtWidgets.QPushButton("Previous Sample")
        self.next_button = QtWidgets.QPushButton("Next Sample")
        self.add_label_button = QtWidgets.QPushButton("Add label")
        self.delete_row_button = QtWidgets.QPushButton("Delete Label")
        self.save_button = QtWidgets.QPushButton("Save Changes")
        
        # Create a horizontal layout for Previous and Next buttons

        nav_buttons_layout = QtWidgets.QHBoxLayout()
        nav_buttons_layout.addWidget(self.prev_button)  # Add Previous button to the left
        nav_buttons_layout.addWidget(self.next_button)  # Add Next button to the right
        layout.addLayout(text_messages)
        layout.addLayout(nav_buttons_layout)
        layout.addWidget(self.add_label_button)
        layout.addWidget(self.delete_row_button)
        layout.addWidget(self.save_button)
        
        # Table for label editing
        self.table = QtWidgets.QTableWidget()
        layout.addWidget(self.table)
        
        # Set up buttons
        self.prev_button.clicked.connect(self.prev_sample)
        self.next_button.clicked.connect(self.next_sample)
        self.add_label_button.clicked.connect(self.add_label)
        self.delete_row_button.clicked.connect(self.mark_row_to_delete)
        self.save_button.clicked.connect(self.save_changes)
        self.setLayout(layout)

        # Load first sample
        self.load_sample()

    def display_generated_image(self):
        # Assume `show_2D` is a method that generates and returns the image
        generatedImage=self.viewer.show_2D()
        self.image_window.display_image(generatedImage)
        self.image_window.show()

    def load_sample(self):
    #Loads the current sample from the dataset and updates the viewer and label table."""
        try:
            P2, V2C, points, image, labels, label_names, labelsOG, name, label_path = self.dataset[self.current_index]
            self.file_label.setText(f"Editing File: {name}")
            self.label_path_of_current_sample=label_path
            print(f"This is the path of the current sample:{self.label_path_of_current_sample}")
            if labels is not None:
                ids = list(range(len(labels)))
            # Debug: Print the shapes of the data components
            """ print(f"P2 shape: {P2.shape}")
            print(f"V2C shape: {V2C.shape}")
            print(f"Points shape: {points.shape}")
            print(f"Image shape: {image.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Label names count: {len(label_names)}") """
            
            # Check if labels are empty and populate the table accordingly
            if labelsOG is None or len(labelsOG) == 0 or len(label_names) == 0:
                self.populate_label_table([], [])
                self.empty_file_message.setText(f"No labels found")
                self.display_sample(P2,V2C, points, image, labels, label_names)
            else:
                self.populate_label_table(labelsOG, label_names)
                self.empty_file_message.setText(f"")
                self.display_sample(P2,V2C, points, image, labels, label_names)
            
        except Exception as e:
            #QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load sample: {e}")
            print(self, "Error", f"Failed to load sample: {e}")

    def display_sample(self, P2,V2C, points, image, labels, label_names ):
        """Displays the current sample in the viewer."""
        ids = list(range(len(labels)))
        self.viewer.add_points(points[:,0:3],
               radius = 2,
               color = (150,150,150),
               scatter_filed = points[:,2],
               alpha=1,
               del_after_show = True,
               add_to_3D_scene = True,
               add_to_2D_scene = True,
               color_map_name = "rainbow")
        self.viewer.add_3D_boxes(
                 #boxes=boxes[:,0:7],
                 labels,
                 ids=ids,
                 box_info=label_names,
                 color="blue",
                 add_to_3D_scene=True,
                 mesh_alpha = 0.3,
                 show_corner_spheres = True,
                 corner_spheres_alpha = 1,
                 corner_spheres_radius=0.1,
                 show_heading = True,
                 heading_scale = 1,
                 show_lines = True,
                 line_width = 2,
                 line_alpha = 1,
                 show_ids = True,
                 show_box_info=True,
                 del_after_show=True,
                 add_to_2D_scene=True,
                 caption_size=(0.3,0.05),#(width, height), it is not the size of the font, it is actually the size of a imaginary box where they text fits, that's why a same size can give different fonts according to how long the word is
                 )
        self.viewer.add_image(image)
        self.viewer.set_extrinsic_mat(V2C)
        self.viewer.set_intrinsic_mat(P2)
        self.display_generated_image()
        self.viewer.show_3D_BEV()
        #for lateral right view
        self.viewerR.add_points(points[:,0:3],
               radius = 2,
               color = (150,150,150),
               scatter_filed = points[:,2],
               alpha=1,
               del_after_show = True,
               add_to_3D_scene = True,
               add_to_2D_scene = True,
               color_map_name = "rainbow")
        self.viewerR.add_3D_boxes(
                 #boxes=boxes[:,0:7],
                 #ids=ids,
                 labels,
                 ids=ids,
                 box_info=label_names,
                 color="blue",
                 add_to_3D_scene=True,
                 mesh_alpha = 0.3,
                 show_corner_spheres = True,
                 corner_spheres_alpha = 1,
                 corner_spheres_radius=0.1,
                 show_heading = True,
                 heading_scale = 1,
                 show_lines = True,
                 line_width = 2,
                 line_alpha = 1,
                 show_ids = True,
                 show_box_info=True,
                 del_after_show=True,
                 add_to_2D_scene=True,
                 caption_size=(0.3,0.05),#(width, height), it is not the size of the font, it is actually the size of a imaginary box where they text fits, that's why a same size can give different fonts according to how long the word is
                 )
        self.viewerR.set_extrinsic_mat(V2C)
        self.viewerR.set_intrinsic_mat(P2)
        self.viewerR.show_3D_right_lateral()
        
        #for lateral left view
        self.viewerL.add_points(points[:,0:3],
               radius = 2,
               color = (150,150,150),
               scatter_filed = points[:,2],
               alpha=1,
               del_after_show = True,
               add_to_3D_scene = True,
               add_to_2D_scene = True,
               color_map_name = "rainbow")
        self.viewerL.add_3D_boxes(
                 #boxes=boxes[:,0:7],
                 #ids=ids,
                 labels,
                 ids=ids,
                 box_info=label_names,
                 color="blue",
                 add_to_3D_scene=True,
                 mesh_alpha = 0.3,
                 show_corner_spheres = True,
                 corner_spheres_alpha = 1,
                 corner_spheres_radius=0.1,
                 show_heading = True,
                 heading_scale = 1,
                 show_lines = True,
                 line_width = 2,
                 line_alpha = 1,
                 show_ids = True,
                 show_box_info=True,
                 del_after_show=True,
                 add_to_2D_scene=True,
                 caption_size=(0.3,0.05),#(width, height), it is not the size of the font, it is actually the size of a imaginary box where they text fits, that's why a same size can give different fonts according to how long the word is
                 )
        self.viewerL.set_extrinsic_mat(V2C)
        self.viewerL.set_intrinsic_mat(P2)
        self.viewerL.show_3D_left_lateral()


    def populate_label_table(self, labelsOG, label_names):
        """Populates the label table with the labels of the current sample."""
        self.table.clear()
        if labelsOG is None or len(labelsOG) == 0:  # No labels case
            self.table.setRowCount(0)
            self.table.setColumnCount(8)
            self.table.setHorizontalHeaderLabels(["Class", "Height", "Width", "Length", "X", "Y", "Z", "Rotation"])
            #self.table.setItem(0, 0, QtWidgets.QTableWidgetItem("No labels found"))
        else:
            self.table.setRowCount(len(labelsOG))
            self.table.setVerticalHeaderLabels([str(i) for i in range(len(labelsOG))])
            self.table.setColumnCount(len(labelsOG[0]) + 1)#one extra column for the class name abd the other one for the ID
            self.table.setHorizontalHeaderLabels(["Class", "Height", "Width", "Length", "X", "Y", "Z", "Rotation"])

            for i, (label_name, label_data) in enumerate(zip(label_names, labelsOG)):
                self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(label_name))
                for j, value in enumerate(label_data):
                    item = QtWidgets.QTableWidgetItem(str(value))
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
                    self.table.setItem(i, j + 1, item)

        self.table.resizeColumnsToContents()

    def next_sample(self):
        """Loads the next sample."""
        if self.current_index < len(self.dataset) - 1:
            self.current_index += 1
            self.load_sample()
        else:
            QtWidgets.QMessageBox.information(self, "Info", "This is the last sample.")

    def prev_sample(self):
        """Loads the previous sample."""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_sample()
        else:
            QtWidgets.QMessageBox.information(self, "Info", "This is the first sample.")

    def add_label(self):
        """Add a new row at the end of the table."""
        row_count = self.table.rowCount()
        self.table.insertRow(row_count)
        # Initialize new row items with default empty values
        for col in range(self.table.columnCount()):
            item = QtWidgets.QTableWidgetItem("")
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
            self.table.setItem(row_count, col, item)

    def mark_row_to_delete(self):
        """Mark the selected row for deletion."""
        selected_rows = self.table.selectionModel().selectedRows()
        
        # Check if only a single cell is selected instead of a whole row
        if not selected_rows:
            selected_indexes = self.table.selectionModel().selectedIndexes()
            if selected_indexes:
                # If only a single cell is selected, get its row and treat it as selected
                row = selected_indexes[0].row()
                self.rows_to_delete.add(row)
                # Select the entire row programmatically to visually highlight it
                self.table.selectRow(row)
            else:
                QtWidgets.QMessageBox.information(self, "Select Row", "Please select a row or cell to delete.")
        else:
            for index in selected_rows:
                self.rows_to_delete.add(index.row())

    def save_changes(self):
        """Save the current table data back to the original text file."""
        with open(self.label_path_of_current_sample, "w") as file:
            for row in range(self.table.rowCount()):
                if row in self.rows_to_delete:
                    continue  # Skip rows marked for deletion
                row_data = []
                for col in range(self.table.columnCount()):
                    item = self.table.item(row, col)
                    if item is not None:
                        row_data.append(item.text())
                if row_data:
                    file.write(" ".join(row_data) + "\n")
        self.rows_to_delete.clear()  # Clear deletion markers after save
        self.load_sample() 
        QtWidgets.QMessageBox.information(self, "Saved", "Changes saved successfully.")
        

def main():
    app = QtWidgets.QApplication(sys.argv)
    dataset_path = "/home/eorozco/annotationTool/dataForViewing/dataset1"
    label_path = "/home/eorozco/annotationTool/dataForViewing/dataset1/labels_2"
    editor = LabelEditor(dataset_path, label_path)
    editor.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

