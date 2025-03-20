mport mlflow
import os, sys
import json
import datetime
import time
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

import cv2 as cv
import pandas as pd
import math

from json_helper import log_print, read_json
from dataset import Template_Dataset

# Check if torchviz is available
try:
    from torchviz import make_dot
    TORCHVIZ_AVAILABLE = True
except ImportError:
    TORCHVIZ_AVAILABLE = False
    log_print("torchviz not available, model architecture visualization will be disabled")


# ***Don't create instance of model directly, this is created in the trainer class
# The Model class inherits from the pytorch model class, and defines the layout of
# the neural network and the forward function of the network.
class __Model__(nn.Module):
    def __init__(self, dataset):
        super(__Model__, self).__init__()

        ### *** START MODEL SPECIFIC *** ###
        # Define the layout of your neural network layers here
        self.xyLayer = nn.Sequential(
            nn.Linear(2, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU()
        )

        self.uvLayer = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 192),
            nn.ReLU(),
        )

        # fc stands for fully-connected
        self.fc = nn.Sequential(
            nn.Linear(192 + 256, 1000),
            nn.ReLU(),
            nn.Linear(1000, 800),
            nn.ReLU(),
            nn.Linear(800, 600),
            nn.ReLU(),
            nn.Linear(600, 3),
            nn.Sigmoid(),
        )

        # These are the parameters needed in the forward function that come from the dataset
        self.pixelPerHogelX = dataset.width / dataset.resPerHogelX
        self.pixelPerHogelY = dataset.height / dataset.resPerHogelY
        self.hResX = dataset.resPerHogelX
        self.hResY = dataset.resPerHogelY
        self.len = dataset.index.shape[0]
        self.width = dataset.width

        ### *** END MODEL SPECIFIC *** ###

    # the forward function, which is what transforms input data into output data at each layer of the
    # network
    def forward(self, index):

        ### *** START MODEL SPECIFIC *** ###
        # The forward function depends on the data being processed
        # and its layout.

        # x,y position in Image

        col = index % self.width
        row = torch.div(index, self.width, rounding_mode="floor")

        # What hogel the pixel is in

        hx = torch.div(col, self.pixelPerHogelX, rounding_mode="floor")
        hy = torch.div(row, self.pixelPerHogelY, rounding_mode="floor")

        # Normalizing data

        hx = torch.div(hx, self.hResX)
        hy = torch.div(hy, self.hResY)

        # Viewing angle of the hogel
        # first gets the pixel location in the hogel with % then divides by the number of pixels in the hogel to
        # Normalizing data

        u = torch.div(col % self.pixelPerHogelX, self.pixelPerHogelX)
        v = torch.div(row % self.pixelPerHogelY, self.pixelPerHogelY)

        # Combining x,y and u,v to allow input into network

        xy = torch.cat((hx, hy), 0)
        uv = torch.cat((u, v), 0)

        # Align into format required for the neural network

        xy = xy.view(2, -1)
        xy = xy.transpose(0, 1)

        # Align into format required for the neural network

        uv = uv.view(2, -1)
        uv = uv.transpose(0, 1)
        xy = self.xyLayer(xy)
        uv = self.uvLayer(uv)
        index = self.fc(torch.cat((xy, uv), 1))

        return index
        ### *** END MODEL SPECIFIC *** ###


class MLflowArtifactLogger:
    """Helper class to manage MLflow artifact logging for holographic models"""
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.model = trainer.model
        self.dataset = trainer.dataset
        self.training_params = trainer.training_params
        self.loss_history = []
        self.epoch_times = []
        self.artifacts_dir = "artifacts"
        
        # Create artifacts directory if it doesn't exist
        if not os.path.exists(self.artifacts_dir):
            os.makedirs(self.artifacts_dir)
    
    def log_model_summary(self):
        """Log a text summary of the model architecture"""
        try:
            # Create a text summary of the model
            model_summary = str(self.model)
            summary_path = os.path.join(self.artifacts_dir, "model_summary.txt")
            
            with open(summary_path, 'w') as f:
                f.write(model_summary)
            
            # Log to MLflow
            mlflow.log_artifact(summary_path)
            log_print("Logged model summary to MLflow")
            
            # Create visualization if torchviz is available
            if TORCHVIZ_AVAILABLE:
                # Create a model visualization using torchviz
                sample_index = torch.tensor([0]).to(self.trainer.device)
                y = self.model(sample_index)
                
                # Create the graph visualization
                dot = make_dot(y, params=dict(list(self.model.named_parameters())))
                dot.format = 'png'
                
                # Save the visualization
                arch_path = os.path.join(self.artifacts_dir, "model_architecture")
                dot.render(arch_path)
                
                # Log to MLflow
                mlflow.log_artifact(arch_path + ".png")
                log_print("Logged model architecture visualization to MLflow")
        except Exception as e:
            log_print(f"Failed to log model summary: {e}")
    
    def log_hogel_structure(self):
        """Creates and logs a visualization of the hogel structure"""
        try:
            # Create a visualization of the hogel structure
            plt.figure(figsize=(10, 8))
            
            # Calculate hogel dimensions
            hogel_width = self.dataset.width / self.dataset.resPerHogelX
            hogel_height = self.dataset.height / self.dataset.resPerHogelY
            
            # Plot hogel grid
            for x in range(self.dataset.resPerHogelX + 1):
                plt.axvline(x * hogel_width, color='white', alpha=0.7)
            
            for y in range(self.dataset.resPerHogelY + 1):
                plt.axhline(y * hogel_height, color='white', alpha=0.7)
            
            # Add labels and title
            plt.title(f"Holographic Display Structure - {self.dataset.resPerHogelX}x{self.dataset.resPerHogelY} Hogels")
            plt.xlim(0, self.dataset.width)
            plt.ylim(self.dataset.height, 0)  # Invert y-axis for image coordinates
            plt.xlabel(f"Width: {self.dataset.width} pixels")
            plt.ylabel(f"Height: {self.dataset.height} pixels")
            
            # Add text annotations for some hogels
            for i in range(min(9, self.dataset.resPerHogelX * self.dataset.resPerHogelY)):
                hogel_x = i % self.dataset.resPerHogelX
                hogel_y = i // self.dataset.resPerHogelX
                
                center_x = (hogel_x + 0.5) * hogel_width
                center_y = (hogel_y + 0.5) * hogel_height
                
                plt.text(center_x, center_y, f"H({hogel_x},{hogel_y})", 
                       color='yellow', ha='center', va='center', fontsize=10)
            
            # Set background color to match typical display
            plt.gca().set_facecolor('black')
            
            # Save the visualization
            hogel_viz_path = os.path.join(self.artifacts_dir, "hogel_structure.png")
            plt.savefig(hogel_viz_path, facecolor='black')
            plt.close()
            
            # Log to MLflow
            mlflow.log_artifact(hogel_viz_path)
            log_print("Logged hogel structure visualization to MLflow")
        except Exception as e:
            log_print(f"Failed to log hogel structure: {e}")
    
    def log_config_file(self):
        """Logs the model configuration file as an artifact"""
        try:
            # Save the config to a file
            config_path = os.path.join(self.artifacts_dir, "model_config.json")
            with open(config_path, 'w') as f:
                json.dump(self.training_params, f, indent=4)
            
            # Log to MLflow
            mlflow.log_artifact(config_path)
            log_print("Logged model configuration to MLflow")
        except Exception as e:
            log_print(f"Failed to log configuration file: {e}")
    
    def log_training_plots(self):
        """Creates and logs plots visualizing the training progress"""
        try:
            # Set seaborn style
            sns.set_style("darkgrid")
            
            # Plot training loss
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(self.loss_history)), self.loss_history, marker='o', linewidth=2)
            plt.title('Training Loss Over Time', fontsize=14)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.grid(True)
            
            # Save the plot
            loss_plot_path = os.path.join(self.artifacts_dir, "training_loss.png")
            plt.savefig(loss_plot_path)
            plt.close()
            
            # Plot epoch times
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(self.epoch_times)), self.epoch_times, marker='o', linewidth=2, color='green')
            plt.title('Training Time Per Epoch', fontsize=14)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Time (seconds)', fontsize=12)
            plt.grid(True)
            
            # Save the plot
            time_plot_path = os.path.join(self.artifacts_dir, "epoch_times.png")
            plt.savefig(time_plot_path)
            plt.close()
            
            # Log to MLflow
            mlflow.log_artifact(loss_plot_path)
            mlflow.log_artifact(time_plot_path)
            log_print("Logged training plots to MLflow")
        except Exception as e:
            log_print(f"Failed to log training plots: {e}")
    
    def log_sample_predictions(self, num_samples=5):
        """Generates and logs visualizations of sample predictions"""
        try:
            # Set model to evaluation mode
            self.model.eval()
            
            # Sample some indices
            indices = torch.from_numpy(
                np.random.choice(self.dataset.len, min(num_samples, self.dataset.len), replace=False)
            ).to(self.trainer.device)
            
            # Get predictions
            with torch.no_grad():
                pred_rgb = self.model(indices).cpu().numpy()
            
            # Get ground truth
            true_rgb = np.array([self.dataset[idx.item()][1].numpy() for idx in indices.cpu()])
            
            # Create visualization
            fig, axes = plt.subplots(num_samples, 2, figsize=(6, 2*num_samples))
            
            # If only one sample, axes won't be 2D
            if num_samples == 1:
                axes = np.array([axes])
                
            for i in range(min(num_samples, len(indices))):
                # Create color patches for visualization
                true_patch = np.ones((50, 50, 3)) * true_rgb[i].reshape(1, 1, 3)
                pred_patch = np.ones((50, 50, 3)) * pred_rgb[i].reshape(1, 1, 3)
                
                # Plot ground truth
                axes[i, 0].imshow(true_patch)
                axes[i, 0].set_title(f"Ground Truth RGB\n{true_rgb[i]}")
                axes[i, 0].axis('off')
                
                # Plot prediction
                axes[i, 1].imshow(pred_patch)
                axes[i, 1].set_title(f"Predicted RGB\n{pred_rgb[i]}")
                axes[i, 1].axis('off')
            
            plt.tight_layout()
            
            # Save the visualization
            pred_path = os.path.join(self.artifacts_dir, "sample_predictions.png")
            plt.savefig(pred_path)
            plt.close()
            
            # Log to MLflow
            mlflow.log_artifact(pred_path)
            log_print("Logged sample predictions to MLflow")
            
            # Set model back to training mode
            self.model.train()
        except Exception as e:
            log_print(f"Failed to log sample predictions: {e}")
    
    def log_checkpoint_as_artifact(self, checkpoint_path):
        """Logs the model checkpoint as an MLflow artifact"""
        try:
            mlflow.log_artifact(checkpoint_path)
            log_print(f"Logged model checkpoint to MLflow: {checkpoint_path}")
        except Exception as e:
            log_print(f"Failed to log model checkpoint: {e}")
    
    def log_dataset_sample(self):
        """Log a visualization of the dataset structure and sample"""
        try:
            # Get the first few samples from the dataset
            num_samples = min(5, self.dataset.len)
            
            # Create a figure to visualize the samples
            fig, axs = plt.subplots(1, num_samples, figsize=(15, 3))
            
            for i in range(num_samples):
                _, rgb_values = self.dataset[i]
                
                # Create a color patch
                rgb_patch = np.ones((50, 50, 3)) * rgb_values.numpy().reshape(1, 1, 3)
                
                if num_samples > 1:
                    axs[i].imshow(rgb_patch)
                    axs[i].set_title(f"Sample {i}\nRGB: {rgb_values.numpy()}")
                    axs[i].axis('off')
                else:
                    axs.imshow(rgb_patch)
                    axs.set_title(f"Sample {i}\nRGB: {rgb_values.numpy()}")
                    axs.axis('off')
            
            plt.tight_layout()
            
            # Save the visualization
            dataset_viz_path = os.path.join(self.artifacts_dir, "dataset_samples.png")
            plt.savefig(dataset_viz_path)
            plt.close()
            
            # Get dataframe structure 
            df_sample = self.dataset.df.head(10)
            df_path = os.path.join(self.artifacts_dir, "dataframe_sample.csv")
            df_sample.to_csv(df_path)
            
            # Log artifacts to MLflow
            mlflow.log_artifact(dataset_viz_path)
            mlflow.log_artifact(df_path)
            log_print("Logged dataset sample to MLflow")
        except Exception as e:
            log_print(f"Failed to log dataset sample: {e}")


# The Trainer class is used to define and train a pytorch model.
class Trainer:
    def __init__(self, training_params, logToMlFlow=True):
        self.training_params = training_params  # a dictionary of parameters from the JSON file for training
        self.run_name = None  # The run name gets set when the training is initialized (see logInitialize())
        self.logToMlFlow = logToMlFlow  # whether or not to log results to MLFlow

        self.dataset = Template_Dataset(
            self.training_params
        )  # the dataset to train the NN on
        self.createModel()

    # Create the pytorch model, which depends on the number of GPUs available for training
    def createModel(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # log the number of GPUs available
        log_print("Using", torch.cuda.device_count(), "GPUs!")

        # if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            # use the DataParallel class
            self.model = nn.DataParallel(__Model__(self.dataset)).to(self.device)
            log_print("multiple gpus")
        else:
            # otherwise, just a basic model
            self.model = __Model__(self.dataset).to(self.device)
            log_print("single gpu")

    # Any code that needs to be run before the training begins goes here.
    # Right now it just sets up MLFlow to get it ready for logging.
    def initializeTraining(self):
        self.logInitialize()

    # The function that does the actual training.
    def train(self):
        ### *** START MODEL SPECIFIC *** ###
        # This code defines the criterion, optimizer, and dataLoader for the training
        criterion = torch.nn.MSELoss()
        optimizer = Adam(
            self.model.parameters(),
            lr=self.training_params.get("modelParams").get("lr"),
            betas=self.training_params.get("modelParams").get("betas"),
        )

        dataLoader = DataLoader(
            dataset=self.dataset,
            batch_size=self.training_params.get("modelParams").get("batch_size"),
            num_workers=2,
            pin_memory=True,
        )
        ### *** END MODEL SPECIFIC *** ###

        # get the number of epochs to run from the modelParams from the JSON file
        epochs = int(self.training_params.get("modelParams").get("epochs"))

        # call the trainEpoch function once for each epoch
        for epoch_index in range(epochs):
            try:
                self.trainEpoch(epoch_index, dataLoader, criterion, optimizer, epochs)

            except Exception as e:
                log_print(f"Exception {e}")

    # Initialize the MLflow logger for artifact logging
    def initializeMLflowLogger(self):
        self.mlflow_logger = MLflowArtifactLogger(self)
        
        # Log initial artifacts
        if self.logToMlFlow:
            # Log configuration as artifact
            self.mlflow_logger.log_config_file()
            # Log model architecture
            self.mlflow_logger.log_model_summary()
            # Log dataset sample
            self.mlflow_logger.log_dataset_sample()
            # Log hogel structure visualization
            self.mlflow_logger.log_hogel_structure()

    # This function contains the code that should run every epoch. Also logs
    # the specified parameters to MLFlow.
    def trainEpoch(self, epoch_index, dataLoader, criterion, optimizer, epochs):
        # Start running of each epoch
        epoch_start = time.time()
        epoch_loss = 0.0

        ### *** START MODEL SPECIFIC *** ###
        # Replace this for loop with code that loops through the output
        # of your model, updating any values that you would like to output
        # to mlflow. Here, we are updating the epoch_loss.
        for index, rgb_values in dataLoader:
            index = index.to(self.device)
            rgb_values = rgb_values.to(self.device)

            y_pred = self.model(index)
            loss = criterion(y_pred, rgb_values)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ### *** END MODEL SPECIFIC *** ###

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        ### *** START MODEL SPECIFIC *** ###
        # add metrics to log on every epoch here
        epochMetrics = {
            f"Epoch Training Time": epoch_time,
            f"Epoch Training Loss": epoch_loss,
        }
        ### *** END MODEL SPECIFIC *** ###

        self.logOnEpoch(epoch_index, epochMetrics)
        self.logOnEpochInterval(epoch_index, epoch_loss, epochs)
        
        # Store data for visualizations if using MLflow
        if self.logToMlFlow and hasattr(self, 'mlflow_logger'):
            self.mlflow_logger.loss_history.append(epoch_loss)
            self.mlflow_logger.epoch_times.append(epoch_time)

    # Any code that needs to be run after the training ends goes here.
    # Right now it just calls the function that ends the MLFlow logging.
    def endTraining(self):
        self.logEnd()

    ### START MLFLOW LOGGING METHODS ###

    # This function sets up MLFlow to get it ready for logging
    # parameters during training. It also sets the name of the run
    # to the branch name followed by the current date/time.
    def logInitialize(self):
        if self.logToMlFlow:
            branchName = str(self.training_params.get("gitParams").get("branchName"))
            dateTimeString = str(datetime.datetime.now().strftime("%m_%d_%y-%H_%M"))
            self.run_name = branchName + "-" + dateTimeString
            mlflow.set_tracking_uri("http://10.220.115.68:5000/")
            mlflow.start_run(run_name=self.run_name)
            mlflow.log_params(self.training_params)
            
            # Initialize the MLflow logger for artifacts
            self.initializeMLflowLogger()

    # This function logs each metric in a dictionary of metrics
    # to MLFlow, and is intended to be called every epoch.
    def logOnEpoch(self, epoch_index, metrics):
        if self.logToMlFlow:
            for key in metrics:
                mlflow.log_metric(key, metrics[key], epoch_index)

    # This function logs parameters on a specified epoch interval,
    # and saves the state of the system to a checkpoint file.
    def logOnEpochInterval(self, epoch_index, epoch_loss, epochs):
        if self.logToMlFlow:
            # get the epoch checkpoint logging interval from the JSON file
            log_interval = int(
                self.training_params.get("modelParams").get("logInterval")
            )

            # if this epoch is on the specified inteval, or it's the last epoch
            if (
                epoch_index % log_interval == 0 and epoch_index != 0
            ) or epoch_index == (epochs - 1):
                branchName = str(
                    self.training_params.get("gitParams").get("branchName")
                )

                # create a folder for the current branch
                checkpointPath = "ModelCheckpoints/" + branchName
                if not os.path.exists(checkpointPath):
                    os.makedirs(checkpointPath)

                # the filename is in the format "month_day_year-hour_minute"
                checkpoint_path = os.path.join(
                    checkpointPath,
                    (
                        "epoch_"
                        + str(epoch_index)
                        + "_date_"
                        + datetime.datetime.now().strftime("%m_%d_%y-%H_%M")
                        + ".pt"
                    ),
                )

                ### *** START MODEL SPECIFIC *** ###
                # Any parameters to log to MLFlow on the specified epoch interval go here.
                # Right now, it just logs the path to the checkpoint.
                checkpointParams = {
                    f"Checkpoint Path for Epoch {epoch_index}": checkpoint_path
                }
                for key in checkpointParams:
                    mlflow.log_param(key, checkpointParams[key])
                ### *** END MODEL SPECIFIC *** ###

                # Save the model to the checkpoint
                state_dict = None

                if isinstance(self.model, nn.DataParallel):
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()

                ### *** START MODEL SPECIFIC *** ###
                # Any parameters to save to the checkpoint go here
                torch.save(
                    {
                        "epoch": epoch_index,
                        "model_state_dict": state_dict,
                        "loss": epoch_loss,
                    },
                    checkpoint_path,
                )
                ### *** END MODEL SPECIFIC *** ###
                
                # Add artifact logging
                if hasattr(self, 'mlflow_logger'):
                    # Log checkpoint as artifact
                    self.mlflow_logger.log_checkpoint_as_artifact(checkpoint_path)
                    
                    # Log sample predictions at intervals
                    self.mlflow_logger.log_sample_predictions()

    # This function ends the MLFlow run
    def logEnd(self):
        if self.logToMlFlow:
            # Create and log summary visualizations if logger exists
            if hasattr(self, 'mlflow_logger'):
                self.mlflow_logger.log_training_plots()
            
            # End the MLflow run
            mlflow.end_run()


if __name__ == "__main__":
    try:
        log_print(datetime.datetime.now().strftime("%m_%d_%y-%H_%M"))
        file = open("modelConfig.json")
        # file = open(os.path.join(os.pardir, "modelConfig.json"))
        training_params = json.load(file)
        file.close()

        trainer = Trainer(training_params)
        trainer.initializeTraining()
        trainer.train()
        trainer.endTraining()

    except Exception as err:
        log_print(str(err.with_traceback()))
