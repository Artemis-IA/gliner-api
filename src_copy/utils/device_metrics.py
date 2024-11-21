# utils/device_metrics.py
import psutil
import GPUtil
from codecarbon import EmissionsTracker
from loguru import logger


class DeviceMetricsTracker:
    def __init__(self, project_name: str = "default_project"):
        self.tracker_active = False
        self.emissions_tracker = None
        self.project_name = project_name

    def start_emissions_tracker(self):
        """Start CodeCarbon tracker."""
        if not self.tracker_active:
            try:
                self.emissions_tracker = EmissionsTracker(project_name=self.project_name)
                self.emissions_tracker.start()
                self.tracker_active = True
                logger.info("CodeCarbon tracker started.")
            except Exception as e:
                logger.warning(f"Unable to start CodeCarbon tracker: {e}")
                self.emissions_tracker = None

    def stop_emissions_tracker(self) -> float:
        """Stop CodeCarbon tracker and return the measured emissions."""
        emissions = None
        if self.emissions_tracker and self.tracker_active:
            try:
                emissions = self.emissions_tracker.stop()
                logger.info(f"Carbon emissions recorded: {emissions:.6f} kgCO₂eq")
            except Exception as e:
                logger.warning(f"Error stopping CodeCarbon tracker: {e}")
            finally:
                self.tracker_active = False
        return emissions

    @staticmethod
    def get_system_metrics() -> dict:
        """Retrieve system metrics (CPU, RAM, GPU)."""
        metrics = {
            "cpu_usage_percent": psutil.cpu_percent(),
            "memory_usage_mb": psutil.virtual_memory().used / (1024 * 1024),  # Convert to MB
        }

        # GPU metrics
        gpus = GPUtil.getGPUs()
        if gpus:
            metrics["gpu_memory_usage_mb"] = gpus[0].memoryUsed  # First GPU only
        else:
            metrics["gpu_memory_usage_mb"] = None

        logger.info(f"System metrics: {metrics}")
        return metrics
