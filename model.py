import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
import torch
from torchviz import make_dot
import mlflow

class MLflowLogger:
    def __init__(self, trainer):
        self.trainer = trainer
        self.model = trainer.model
        self.dataset = trainer.dataset
        self.training_params = trainer.training_params
        self.loss_history = []
        self.epoch_times = []
        self.artifacts_dir = "run_artifacts"
        
        # Create artifacts directory if it doesn't exist
        if not os.path.exists(self.artifacts_dir):
            os.makedirs(self.artifacts_dir)
    
    def log_model_architecture(self):
        """Creates and logs a visualization of the model architecture"""
        try:
            # Create a model visualization using torchviz
            # We need a sample input for this
            sample_index = torch.tensor([0]).to(self.trainer.device)
            y = self.model(sample_index)
            
            # Create the graph visualization
            dot = make_dot(y, params=dict(self.model.named_parameters()))
            dot.format = 'png'
            
            # Save the visualization
            arch_path = os.path.join(self.artifacts_dir, "model_architecture.png")
            dot.render(arch_path)
            
            # Log to MLflow
            mlflow.log_artifact(arch_path + ".png")
            print(f"Logged model architecture visualization to MLflow")
        except Exception as e:
            print(f"Failed to log model architecture: {e}")
    
    def log_config_file(self):
        """Logs the model configuration file as an artifact"""
        try:
            # Save the config to a file
            config_path = os.path.join(self.artifacts_dir, "model_config.json")
            with open(config_path, 'w') as f:
                json.dump(self.training_params, f, indent=4)
            
            # Log to MLflow
            mlflow.log_artifact(config_path)
            print(f"Logged model configuration to MLflow")
        except Exception as e:
            print(f"Failed to log configuration file: {e}")
    
    def log_training_plots(self):
        """Creates and logs plots visualizing the training progress"""
        try:
            # Plot training loss
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(self.loss_history)), self.loss_history, marker='o')
            plt.title('Training Loss Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            
            # Save the plot
            loss_plot_path = os.path.join(self.artifacts_dir, "training_loss.png")
            plt.savefig(loss_plot_path)
            plt.close()
            
            # Plot epoch times
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(self.epoch_times)), self.epoch_times, marker='o', color='green')
            plt.title('Training Time Per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Time (seconds)')
            plt.grid(True)
            
            # Save the plot
            time_plot_path = os.path.join(self.artifacts_dir, "epoch_times.png")
            plt.savefig(time_plot_path)
            plt.close()
            
            # Log to MLflow
            mlflow.log_artifact(loss_plot_path)
            mlflow.log_artifact(time_plot_path)
            print(f"Logged training plots to MLflow")
        except Exception as e:
            print(f"Failed to log training plots: {e}")
    
    def log_sample_predictions(self, num_samples=5):
        """Generates and logs visualizations of sample predictions"""
        try:
            # Set model to evaluation mode
            self.model.eval()
            
            # Sample some indices
            indices = torch.tensor(np.random.choice(self.dataset.index.shape[0], num_samples)).to(self.trainer.device)
            
            # Get predictions
            with torch.no_grad():
                predictions = self.model(indices).cpu().numpy()
            
            # Get ground truth
            ground_truth = []
            for idx in indices.cpu().numpy():
                ground_truth.append(self.dataset[idx][1].numpy())
            
            # Create visualization
            fig, axes = plt.subplots(num_samples, 2, figsize=(8, 3*num_samples))
            
            for i in range(num_samples):
                # Plot ground truth
                axes[i, 0].imshow(np.array([ground_truth[i]]).reshape(1, 1, 3))
                axes[i, 0].set_title(f"Sample {i+1}: Ground Truth")
                axes[i, 0].axis('off')
                
                # Plot prediction
                axes[i, 1].imshow(predictions[i].reshape(1, 1, 3))
                axes[i, 1].set_title(f"Sample {i+1}: Prediction")
                axes[i, 1].axis('off')
            
            plt.tight_layout()
            
            # Save the visualization
            pred_path = os.path.join(self.artifacts_dir, "sample_predictions.png")
            plt.savefig(pred_path)
            plt.close()
            
            # Log to MLflow
            mlflow.log_artifact(pred_path)
            print(f"Logged sample predictions to MLflow")
            
            # Set model back to training mode
            self.model.train()
        except Exception as e:
            print(f"Failed to log sample predictions: {e}")
    
    def log_checkpoint_as_artifact(self, checkpoint_path):
        """Logs the model checkpoint as an MLflow artifact"""
        try:
            mlflow.log_artifact(checkpoint_path)
            print(f"Logged model checkpoint to MLflow: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to log model checkpoint: {e}")

# Extended Trainer class method to integrate with MLflowLogger
def enhancedTrainer(Trainer):
    """Extends the Trainer class with enhanced MLflow artifact logging"""
    
    # Create a new method to initialize the MLflow logger
    def initializeMLflowLogger(self):
        self.mlflow_logger = MLflowLogger(self)
        
    # Enhance the logInitialize method
    original_logInitialize = Trainer.logInitialize
    def enhanced_logInitialize(self):
        original_logInitialize(self)
        if self.logToMlFlow:
            # Initialize MLflow logger
            self.initializeMLflowLogger()
            # Log configuration as artifact
            self.mlflow_logger.log_config_file()
            # Log model architecture visualization
            self.mlflow_logger.log_model_architecture()
    
    # Enhance the trainEpoch method
    original_trainEpoch = Trainer.trainEpoch
    def enhanced_trainEpoch(self, epoch_index, dataLoader, criterion, optimizer, epochs):
        epoch_start = time.time()
        result = original_trainEpoch(self, epoch_index, dataLoader, criterion, optimizer, epochs)
        epoch_time = time.time() - epoch_start
        
        if self.logToMlFlow:
            # Store loss and time data for plotting
            self.mlflow_logger.loss_history.append(epoch_loss)
            self.mlflow_logger.epoch_times.append(epoch_time)
        
        return result
    
    # Enhance the logOnEpochInterval method
    original_logOnEpochInterval = Trainer.logOnEpochInterval
    def enhanced_logOnEpochInterval(self, epoch_index, epoch_loss, epochs):
        original_logOnEpochInterval(self, epoch_index, epoch_loss, epochs)
        
        if self.logToMlFlow:
            # Get the log interval
            log_interval = int(self.training_params.get("modelParams").get("logInterval"))
            
            # If this is an interval to log or the last epoch
            if (epoch_index % log_interval == 0 and epoch_index != 0) or epoch_index == (epochs - 1):
                # Get checkpoint path from the original method
                branchName = str(self.training_params.get("gitParams").get("branchName"))
                checkpointPath = "ModelCheckpoints/" + branchName
                checkpoint_filename = (
                    "epoch_"
                    + str(epoch_index)
                    + "_date_"
                    + datetime.datetime.now().strftime("%m_%d_%y-%H_%M")
                    + ".pt"
                )
                checkpoint_path = os.path.join(checkpointPath, checkpoint_filename)
                
                # Log checkpoint as artifact
                self.mlflow_logger.log_checkpoint_as_artifact(checkpoint_path)
                
                # Log sample predictions
                self.mlflow_logger.log_sample_predictions()
    
    # Enhance the logEnd method
    original_logEnd = Trainer.logEnd
    def enhanced_logEnd(self):
        if self.logToMlFlow:
            # Create and log summary plots
            self.mlflow_logger.log_training_plots()
        
        original_logEnd(self)
    
    # Apply the enhanced methods to the Trainer class
    Trainer.initializeMLflowLogger = initializeMLflowLogger
    Trainer.logInitialize = enhanced_logInitialize
    Trainer.trainEpoch = enhanced_trainEpoch
    Trainer.logOnEpochInterval = enhanced_logOnEpochInterval
    Trainer.logEnd = enhanced_logEnd
    
    return Trainer

# Usage example:
# EnhancedTrainer = enhancedTrainer(Trainer)
# trainer = EnhancedTrainer(training_params)
# trainer.initializeTraining()
# trainer.train()
# trainer.endTraining()
