from abc import ABC, abstractmethod

class BaseEnvironment(ABC):
    """
    Abstract base class for environments.
    """

    def __init__(self, config):
        """
        Initialize the environment with a given configuration.

        :param config: Configuration dictionary for the environment.
        """
        self.config = config
        self.initialize_time()
        self.init_env()

    @abstractmethod
    def init_env(self):
        """
        Abstract method to initialize the environment.
        This method should be implemented by each specific environment class.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the environment to its initial state.
        This method should be implemented by each specific environment class.
        """
        pass

    def _reset_internal(self):
        """
        Reset any internal variables.
        """
        self.cur_time = 0
        self.timestep = 0

    @abstractmethod
    def step(self, action):
        """
        Apply an action to the environment and advance to the next state.

        :param action: The action to be applied.
        """
        pass

    @abstractmethod
    def render(self, mode='human'):
        """
        Render the environment.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Perform any necessary cleanup.
        """
        pass
    
    def initialize_time(self):
        """
        Initialize time-related variables.

        :param control_freq: Control frequency.
        """
        self.timestep = 0
        self.cur_time = 0
        
        self._model_timestep = self.config['sim_timestep']
        if self._model_timestep <=0:
            raise ValueError("Invalid simulation timestep defined!")
        
        self._control_freq = self.config['control_freq']
        self._control_timestep = 1. / self._control_freq
        if self._control_timestep <=0:
            raise ValueError("Invalid control timestep defined!")
        
        self._policy_freq = self.config['policy_freq']
        self._policy_timestep = 1. / self._policy_freq
        if self._policy_timestep <=0:
            raise ValueError("Invalid policy timestep defined!")
        
        print("Control timestep: ", self._control_timestep)
        print("Model timestep: ", self._model_timestep)
        print("Policy timestep:", self._policy_timestep)