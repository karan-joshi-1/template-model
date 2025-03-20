# Add these imports at the top of your model.py file, with the other imports
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
try:
    from torchviz import make_dot
    TORCHVIZ_AVAILABLE = True
except ImportError:
    TORCHVIZ_AVAILABLE = False
    log_print("torchviz not available, model architecture visualization will be disabled")

# Add this class after the Trainer class but before if __name__ == "__main__"
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
            
            # Add text labels for some hogels
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

# Now, modify the Trainer class methods:

# Add this method to the Trainer class
def initializeMLflowLogger(self):
    """Initialize the MLflow logger for artifact logging"""
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

# Replace the existing logInitialize method with this enhanced version
def enhanced_logInitialize(self):
    if self.logToMlFlow:
        branchName = str(self.training_params.get("gitParams").get("branchName"))
        dateTimeString = str(datetime.datetime.now().strftime("%m_%d_%y-%H_%M"))
        self.run_name = branchName + "-" + dateTimeString
        mlflow.set_tracking_uri("http://10.220.115.62:5000/")
        mlflow.start_run(run_name=self.run_name)
        mlflow.log_params(self.training_params)
        
        # Initialize the MLflow logger for artifacts
        self.initializeMLflowLogger()

# Replace the existing trainEpoch method with this enhanced version
def enhanced_trainEpoch(self, epoch_index, dataLoader, criterion, optimizer, epochs):
    # Start running of each epoch
    epoch_start = time.time()
    epoch_loss = 0.0

    # Training loop - same as original
    for index, rgb_values in dataLoader:
        index = index.to(self.device)
        rgb_values = rgb_values.to(self.device)

        y_pred = self.model(index)
        loss = criterion(y_pred, rgb_values)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start

    # Metrics to log - same as original
    epochMetrics = {
        f"Epoch Training Time": epoch_time,
        f"Epoch Training Loss": epoch_loss,
    }

    # Original logging calls
    self.logOnEpoch(epoch_index, epochMetrics)
    self.logOnEpochInterval(epoch_index, epoch_loss, epochs)
    
    # Store data for visualizations if using MLflow
    if self.logToMlFlow and hasattr(self, 'mlflow_logger'):
        self.mlflow_logger.loss_history.append(epoch_loss)
        self.mlflow_logger.epoch_times.append(epoch_time)

# Replace the existing logOnEpochInterval method with this enhanced version
def enhanced_logOnEpochInterval(self, epoch_index, epoch_loss, epochs):
    if self.logToMlFlow:
        # Use the existing code for interval determination
        log_interval = int(
            self.training_params.get("modelParams").get("logInterval")
        )

        # if this epoch is on the specified interval, or it's the last epoch
        if (
            epoch_index % log_interval == 0 and epoch_index != 0
        ) or epoch_index == (epochs - 1):
            branchName = str(
                self.training_params.get("gitParams").get("branchName")
            )

            # create a folder for the current branch - same as original
            checkpointPath = "ModelCheckpoints/" + branchName
            if not os.path.exists(checkpointPath):
                os.makedirs(checkpointPath)

            # create checkpoint filename - same as original
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

            # Log parameters - same as original
            checkpointParams = {
                f"Checkpoint Path for Epoch {epoch_index}": checkpoint_path
            }
            for key in checkpointParams:
                mlflow.log_param(key, checkpointParams[key])

            # Save the model state - same as original
            state_dict = None
            if isinstance(self.model, nn.DataParallel):
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()

            # Save checkpoint - same as original
            torch.save(
                {
                    "epoch": epoch_index,
                    "model_state_dict": state_dict,
                    "loss": epoch_loss,
                },
                checkpoint_path,
            )
            
            # Add artifact logging
            if hasattr(self, 'mlflow_logger'):
                # Log checkpoint as artifact
                self.mlflow_logger.log_checkpoint_as_artifact(checkpoint_path)
                
                # Log sample predictions at intervals
                self.mlflow_logger.log_sample_predictions()

# Replace the existing logEnd method with this enhanced version
def enhanced_logEnd(self):
    if self.logToMlFlow:
        # Create and log summary visualizations if logger exists
        if hasattr(self, 'mlflow_logger'):
            self.mlflow_logger.log_training_plots()
        
        # End the MLflow run - same as original
        mlflow.end_run()
