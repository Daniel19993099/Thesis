from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_absolute_error
import numpy as np
from IPython.display import display, clear_output

class PrintLossCallback(Callback):
        def __init__(self, name, interval=1000):
            self.interval = interval
            self.name=name
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.interval == 0: 
                loss = logs.get(self.name)  
                print(f"Epoch {epoch+1}: Loss = {loss:.10f}")

class EarlyStoppingByTrainMAE(Callback):
    def __init__(self,name, threshold=0.05, verbose=1):
        super().__init__()
        self.threshold = threshold
        self.name=name
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        """Stop training if train MAE goes below threshold"""
        if logs is None:
            return

        loss = logs.get(self.name)  # Get training MAE from logs
        if loss is not None and loss <= self.threshold:
            print(f"\n Stopping training: Train reached {loss:.4f}, below threshold {self.threshold} at epoch {epoch+1}. ")
            self.model.stop_training = True

class LiveScatterTwoOutputs(Callback):
    def __init__(self, X_train_nodes,X_test_nodes, X_train_elements,X_test_elements, y_train,y_test, interval=500):
        super().__init__()
        self.X_train_nodes = X_train_nodes
        self.X_train_elements = X_train_elements
        self.y_train = y_train
        self.X_test_nodes = X_test_nodes
        self.X_test_elements = X_test_elements
        self.y_test = y_test
        self.interval = interval  # Update every N epochs
        
        plt.ion()  # Enable interactive mode

        # Set up the figure
        self.fig, self.ax = plt.subplots(1,2,figsize=(12,6))
        self.train_ax = self.ax[0]  # First subplot for training data
        self.test_ax = self.ax[1]   # Second subplot for test data
    

        # Initial scatter plot for train data
        self.train_ax.plot([0, 1], [0, 1], 'r--', label="Ideal")  # y=x reference line
        self.train_scatter = self.train_ax.scatter([], [], alpha=0.5, color="blue", label="Train Predictions")
        self.train_ax.set_xlim(0, 1)
        self.train_ax.set_ylim(0, 1)
        self.train_ax.set_xlabel("Train Data")
        self.train_ax.set_ylabel("Predicted Strength")
        self.train_ax.set_title("Train Predictions vs. True Values")
        self.train_ax.legend()

        # Initial scatter plot for test data
        self.test_ax.plot([0, 1], [0, 1], 'r--', label="Ideal")  # y=x reference line
        self.test_scatter = self.test_ax.scatter([], [], alpha=0.5, color="green", label="Test Predictions")
        self.test_ax.set_xlim(0, 1)
        self.test_ax.set_ylim(0, 1)
        self.test_ax.set_xlabel("Test Data")
        self.test_ax.set_ylabel("Predicted Strength")
        self.test_ax.set_title("Test Predictions vs. True Values")
        self.test_ax.legend()

        plt.show(block=False)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            print(f"Updating scatter plot at epoch {epoch + 1}...")
            #lr = float(self.model.optimizer.lr.numpy())
            #self.learning_rates.append(lr) 

            y_train_pred, _ = self.model.predict([self.X_train_nodes, self.X_train_elements], verbose=0)
            y_test_pred, _ = self.model.predict([self.X_test_nodes, self.X_test_elements], verbose=0)
            

            # Ensure correct shape
            y_train_pred = np.array(y_train_pred).flatten() 
            y_test_pred = np.array(y_test_pred).flatten()
            
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)

            # Clear previous output in Jupyter Notebook
            clear_output(wait=True)
            lr= logs.get("learning_rate")  
            # Update train scatter plot
            self.train_ax.clear()
            self.train_ax.plot([0, 1], [0, 1], 'r--', label="Ideal")
            self.train_ax.scatter(self.y_train.flatten(), y_train_pred, alpha=0.5, color="blue", label="Train Predictions")
            self.train_ax.set_xlim(0, 1)
            self.train_ax.set_ylim(0, 1)
            self.train_ax.set_xlabel("Train Data")
            self.train_ax.set_ylabel("Predicted Strength")
            self.train_ax.set_title(f"Train Predictions{len(self.y_train)} - Epoch {epoch+1}, MAE: {train_mae:.5f}\n Learning Rate: {lr:.5f}")
            self.train_ax.legend()

            # Update test scatter plot
            self.test_ax.clear()
            self.test_ax.plot([0, 1], [0, 1], 'r--', label="Ideal")
            self.test_ax.scatter(self.y_test.flatten(), y_test_pred, alpha=0.5, color="green", label="Test Predictions")
            self.test_ax.set_xlim(0, 1)
            self.test_ax.set_ylim(0, 1)
            self.test_ax.set_xlabel("Test Data")
            self.test_ax.set_ylabel("Predicted Strength")
            self.test_ax.set_title(f"Test Predictions{len(self.y_test)} - Epoch {epoch+1}, MAE: {test_mae:.5f}\n Learning Rate: {lr:.5f}")
            self.test_ax.legend()

            # Redraw the updated figure
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)  # Ensure real-time updates
            display(self.fig)  # Explicitly display the figure

class LiveScatterCNN(Callback):
    def __init__(self, X_train_nodes,X_test_nodes, y_train,y_test, interval=500):
        super().__init__()
        self.X_train_nodes = X_train_nodes
        self.y_train = y_train
        self.X_test_nodes = X_test_nodes
        self.y_test = y_test
        self.interval = interval  # Update every N epochs
        
        plt.ion()  # Enable interactive mode

        # Set up the figure
        self.fig, self.ax = plt.subplots(1,2,figsize=(12,6))
        self.train_ax = self.ax[0]  # First subplot for training data
        self.test_ax = self.ax[1]   # Second subplot for test data
    

        # Initial scatter plot for train data
        self.train_ax.plot([0, 1], [0, 1], 'r--', label="Ideal")  # y=x reference line
        self.train_scatter = self.train_ax.scatter([], [], alpha=0.5, color="blue", label="Train Predictions")
        self.train_ax.set_xlim(0, 1)
        self.train_ax.set_ylim(0, 1)
        self.train_ax.set_xlabel("Train Data")
        self.train_ax.set_ylabel("Predicted Strength")
        self.train_ax.set_title("Train Predictions vs. True Values")
        self.train_ax.legend()

        # Initial scatter plot for test data
        self.test_ax.plot([0, 1], [0, 1], 'r--', label="Ideal")  # y=x reference line
        self.test_scatter = self.test_ax.scatter([], [], alpha=0.5, color="green", label="Test Predictions")
        self.test_ax.set_xlim(0, 1)
        self.test_ax.set_ylim(0, 1)
        self.test_ax.set_xlabel("Test Data")
        self.test_ax.set_ylabel("Predicted Strength")
        self.test_ax.set_title("Test Predictions vs. True Values")
        self.test_ax.legend()

        plt.show(block=False)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            print(f"Updating scatter plot at epoch {epoch + 1}...")
            #lr = float(self.model.optimizer.lr.numpy())
            #self.learning_rates.append(lr) 

            y_train_pred, _ = self.model.predict([self.X_train_nodes], verbose=0)
            y_test_pred, _ = self.model.predict([self.X_test_nodes], verbose=0)
            

            # Ensure correct shape
            y_train_pred = np.array(y_train_pred).flatten() 
            y_test_pred = np.array(y_test_pred).flatten()
            
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)

            # Clear previous output in Jupyter Notebook
            clear_output(wait=True)
            lr= logs.get("learning_rate")  
            # Update train scatter plot
            self.train_ax.clear()
            self.train_ax.plot([0, 1], [0, 1], 'r--', label="Ideal")
            self.train_ax.scatter(self.y_train.flatten(), y_train_pred, alpha=0.5, color="blue", label="Train Predictions")
            self.train_ax.set_xlim(0, 1)
            self.train_ax.set_ylim(0, 1)
            self.train_ax.set_xlabel("Train Data")
            self.train_ax.set_ylabel("Predicted Strength")
            self.train_ax.set_title(f"Train Predictions{len(self.y_train)} - Epoch {epoch+1}, MAE: {train_mae:.5f}\n Learning Rate: {lr:.5f}")
            self.train_ax.legend()

            # Update test scatter plot
            self.test_ax.clear()
            self.test_ax.plot([0, 1], [0, 1], 'r--', label="Ideal")
            self.test_ax.scatter(self.y_test.flatten(), y_test_pred, alpha=0.5, color="green", label="Test Predictions")
            self.test_ax.set_xlim(0, 1)
            self.test_ax.set_ylim(0, 1)
            self.test_ax.set_xlabel("Test Data")
            self.test_ax.set_ylabel("Predicted Strength")
            self.test_ax.set_title(f"Test Predictions{len(self.y_test)} - Epoch {epoch+1}, MAE: {test_mae:.5f}\n Learning Rate: {lr:.5f}")
            self.test_ax.legend()

            # Redraw the updated figure
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)  # Ensure real-time updates
            display(self.fig)  # Explicitly display the figure


class LiveScatterThreeOutputs(Callback):
    def __init__(self, X_train_nodes,X_test_nodes, X_train_elements,X_test_elements, y_train,y_test, interval=500):
        super().__init__()
        self.X_train_nodes = X_train_nodes
        self.X_train_elements = X_train_elements
        self.y_train = y_train
        self.X_test_nodes = X_test_nodes
        self.X_test_elements = X_test_elements
        self.y_test = y_test
        self.interval = interval  # Update every N epochs
        
        plt.ion()  # Enable interactive mode

        # Set up the figure
        self.fig, self.ax = plt.subplots(1,2,figsize=(12,6))
        self.train_ax = self.ax[0]  # First subplot for training data
        self.test_ax = self.ax[1]   # Second subplot for test data
    

        # Initial scatter plot for train data
        self.train_ax.plot([0, 1], [0, 1], 'r--', label="Ideal")  # y=x reference line
        self.train_scatter = self.train_ax.scatter([], [], alpha=0.5, color="blue", label="Train Predictions")
        self.train_ax.set_xlim(0, 1)
        self.train_ax.set_ylim(0, 1)
        self.train_ax.set_xlabel("Train Data")
        self.train_ax.set_ylabel("Predicted Strength")
        self.train_ax.set_title("Train Predictions vs. True Values")
        self.train_ax.legend()

        # Initial scatter plot for test data
        self.test_ax.plot([0, 1], [0, 1], 'r--', label="Ideal")  # y=x reference line
        self.test_scatter = self.test_ax.scatter([], [], alpha=0.5, color="green", label="Test Predictions")
        self.test_ax.set_xlim(0, 1)
        self.test_ax.set_ylim(0, 1)
        self.test_ax.set_xlabel("Test Data")
        self.test_ax.set_ylabel("Predicted Strength")
        self.test_ax.set_title("Test Predictions vs. True Values")
        self.test_ax.legend()

        plt.show(block=False)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            print(f"Updating scatter plot at epoch {epoch + 1}...")
            #lr = float(self.model.optimizer.lr.numpy())
            #self.learning_rates.append(lr) 

            y_train_pred, _ ,_= self.model.predict([self.X_train_nodes, self.X_train_elements], verbose=0)
            y_test_pred, _ ,_= self.model.predict([self.X_test_nodes, self.X_test_elements], verbose=0)
            

            # Ensure correct shape
            y_train_pred = np.array(y_train_pred).flatten() 
            y_test_pred = np.array(y_test_pred).flatten()
            
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)

            # Clear previous output in Jupyter Notebook
            clear_output(wait=True)
            lr= logs.get("learning_rate")  
            # Update train scatter plot
            self.train_ax.clear()
            self.train_ax.plot([0, 1], [0, 1], 'r--', label="Ideal")
            self.train_ax.scatter(self.y_train.flatten(), y_train_pred, alpha=0.5, color="blue", label="Train Predictions")
            self.train_ax.set_xlim(0, 1)
            self.train_ax.set_ylim(0, 1)
            self.train_ax.set_xlabel("Train Data")
            self.train_ax.set_ylabel("Predicted Strength")
            self.train_ax.set_title(f"Train Predictions{len(self.y_train)} - Epoch {epoch+1}, MAE: {train_mae:.5f}\n Learning Rate: {lr:.5f}")
            self.train_ax.legend()

            # Update test scatter plot
            self.test_ax.clear()
            self.test_ax.plot([0, 1], [0, 1], 'r--', label="Ideal")
            self.test_ax.scatter(self.y_test.flatten(), y_test_pred, alpha=0.5, color="green", label="Test Predictions")
            self.test_ax.set_xlim(0, 1)
            self.test_ax.set_ylim(0, 1)
            self.test_ax.set_xlabel("Test Data")
            self.test_ax.set_ylabel("Predicted Strength")
            self.test_ax.set_title(f"Test Predictions{len(self.y_test)} - Epoch {epoch+1}, MAE: {test_mae:.5f}\n Learning Rate: {lr:.5f}")
            self.test_ax.legend()

            # Redraw the updated figure
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)  # Ensure real-time updates
            display(self.fig)  # Explicitly display the figure

