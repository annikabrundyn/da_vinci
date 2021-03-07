from typing import Callable


class LossRegistry:
    """This registry is used to dynamically track all available losses.

    Attributes:
        registry (dict): Dynamic registry mapping loss names to loss modules
    """

    registry = {}

    @classmethod
    def register(cls, loss_name) -> Callable:
        """Register the loss_name as a key mapping to the loss CustomLoss

        Args:
            loss_name (str):

        Returns:
            Callable: inner_wrapper() wrapping loss nn.Module
        """

        def inner_wrapper(wrapped_class) -> Callable:
            """Summary

            Args:
                wrapped_class (nn.Module): Loss class
            Returns:
                Callable: Loss class
            """
            assert loss_name not in cls.registry
            cls.registry[loss_name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get_registry(cls) -> dict:
        return cls.registry
