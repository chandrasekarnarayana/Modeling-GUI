from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox

class RandomForestDialog(QDialog):
    def __init__(self, parent=None):
        super(RandomForestDialog, self).__init__(parent)
        self.setWindowTitle("Random Forest Parameters")

        layout = QVBoxLayout()

        # Number of estimators
        self.n_estimators_label = QLabel("Number of Estimators:")
        self.n_estimators_input = QSpinBox()
        self.n_estimators_input.setMinimum(1)
        self.n_estimators_input.setMaximum(1000)
        self.n_estimators_input.setValue(100)
        layout.addWidget(self.n_estimators_label)
        layout.addWidget(self.n_estimators_input)

        # Max Depth
        self.max_depth_label = QLabel("Max Depth (Optional):")
        self.max_depth_input = QSpinBox()
        self.max_depth_input.setMinimum(1)
        self.max_depth_input.setMaximum(100)
        self.max_depth_input.setValue(10)
        layout.addWidget(self.max_depth_label)
        layout.addWidget(self.max_depth_input)

        # OK/Cancel Buttons
        buttons_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(self.ok_button)
        buttons_layout.addWidget(self.cancel_button)
        layout.addLayout(buttons_layout)

        self.setLayout(layout)

    @property
    def n_estimators(self):
        return self.n_estimators_input.value()

    @property
    def max_depth(self):
        return self.max_depth_input.value()

class GradientBoostDialog(QDialog):
    def __init__(self, parent=None):
        super(GradientBoostDialog, self).__init__(parent)
        self.setWindowTitle("Gradient Boost Parameters")

        layout = QVBoxLayout()

        # Number of estimators
        self.n_estimators_label = QLabel("Number of Estimators:")
        self.n_estimators_input = QSpinBox()
        self.n_estimators_input.setMinimum(1)
        self.n_estimators_input.setMaximum(1000)
        self.n_estimators_input.setValue(100)
        layout.addWidget(self.n_estimators_label)
        layout.addWidget(self.n_estimators_input)

        # Learning rate
        self.learning_rate_label = QLabel("Learning Rate:")
        self.learning_rate_input = QDoubleSpinBox()
        self.learning_rate_input.setMinimum(0.001)
        self.learning_rate_input.setMaximum(1.0)
        self.learning_rate_input.setSingleStep(0.01)
        self.learning_rate_input.setValue(0.1)
        layout.addWidget(self.learning_rate_label)
        layout.addWidget(self.learning_rate_input)

        # Max Depth
        self.max_depth_label = QLabel("Max Depth (Optional):")
        self.max_depth_input = QSpinBox()
        self.max_depth_input.setMinimum(1)
        self.max_depth_input.setMaximum(100)
        self.max_depth_input.setValue(3)
        layout.addWidget(self.max_depth_label)
        layout.addWidget(self.max_depth_input)

        # OK/Cancel Buttons
        buttons_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(self.ok_button)
        buttons_layout.addWidget(self.cancel_button)
        layout.addLayout(buttons_layout)

        self.setLayout(layout)

    @property
    def n_estimators(self):
        return self.n_estimators_input.value()

    @property
    def learning_rate(self):
        return self.learning_rate_input.value()

    @property
    def max_depth(self):
        return self.max_depth_input.value()

class KMeansDialog(QDialog):
    def __init__(self, parent=None):
        super(KMeansDialog, self).__init__(parent)
        self.setWindowTitle("KMeans Parameters")

        layout = QVBoxLayout()

        # Number of clusters
        self.n_clusters_label = QLabel("Number of Clusters:")
        self.n_clusters_input = QSpinBox()
        self.n_clusters_input.setMinimum(1)
        self.n_clusters_input.setMaximum(100)
        self.n_clusters_input.setValue(3)
        layout.addWidget(self.n_clusters_label)
        layout.addWidget(self.n_clusters_input)

        # OK/Cancel Buttons
        buttons_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(self.ok_button)
        buttons_layout.addWidget(self.cancel_button)
        layout.addLayout(buttons_layout)

        self.setLayout(layout)

    @property
    def n_clusters(self):
        return self.n_clusters_input.value()

